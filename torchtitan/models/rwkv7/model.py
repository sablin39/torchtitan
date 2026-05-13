# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict, Sequential
from torchtitan.tools.logging import logger


LayerNorm = Module.from_nn_module(nn.LayerNorm)
GroupNorm = Module.from_nn_module(nn.GroupNorm)
Tanh = Module.from_nn_module(nn.Tanh)
Sigmoid = Module.from_nn_module(nn.Sigmoid)
Identity = Module.from_nn_module(nn.Identity)


def _zero_(param: torch.Tensor) -> None:
    nn.init.zeros_(param)


def _ones_(param: torch.Tensor) -> None:
    nn.init.ones_(param)


def _orthogonal_(param: torch.Tensor, gain: float = 1.0) -> None:
    original_dtype = param.dtype
    if isinstance(param, DTensor):
        value = torch.empty(param.shape, device=param.device, dtype=torch.float32)
    else:
        value = torch.empty_like(param, dtype=torch.float32)
    nn.init.orthogonal_(value, gain=gain)
    with torch.no_grad():
        _copy_tensor_(param, value.to(original_dtype))


def _copy_tensor_(dst: torch.Tensor, src: torch.Tensor) -> None:
    src = src.to(device=dst.device, dtype=dst.dtype)
    if isinstance(dst, DTensor) and not isinstance(src, DTensor):
        src = distribute_tensor(src, dst.device_mesh, list(dst.placements))
    dst.copy_(src)


def _embedding_init(param: torch.Tensor) -> None:
    nn.init.normal_(param, std=1.0)


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": _zero_,
    }


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    param_init: dict[str, Callable] | None = None,
) -> Linear:
    if param_init is None:
        param_init = {"weight": _zero_}
        if bias:
            param_init["bias"] = _zero_
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=param_init,
    ).build()


def _sqrelu(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


def _reshape_heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    batch, seq_len, width = x.shape
    return x.view(batch, seq_len, width // head_dim, head_dim)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, seq_len, n_heads, head_dim = x.shape
    return x.reshape(batch, seq_len, n_heads * head_dim)


class _FLAOps:
    def __init__(self) -> None:
        try:
            from fla.modules.l2norm import l2_norm
            from fla.modules.token_shift import token_shift
            from fla.modules.token_shift_cp import token_shift_cp
            from fla.ops.cp import build_cp_context
            from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
            from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
            from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
            from fla.ops.rwkv7.gate_output_correction import gate_output_correction
        except ImportError as exc:
            raise ImportError(
                "RWKV7 requires flash-linear-attention (FLA). Install a version "
                "that provides fla.ops.cp, fla.modules.token_shift_cp, and "
                "fla.ops.generalized_delta_rule.dplr."
            ) from exc

        self.l2_norm = l2_norm
        self.token_shift = token_shift
        self.token_shift_cp = token_shift_cp
        self.build_cp_context = build_cp_context
        self.chunk_dplr_delta_rule = chunk_dplr_delta_rule
        self.fused_addcmul_rwkv7 = fused_addcmul_rwkv7
        self.fused_k_rwkv7 = fused_k_rwkv7
        self.gate_output_correction = gate_output_correction


_fla_ops: _FLAOps | None = None


def _require_fla_ops() -> _FLAOps:
    global _fla_ops
    if _fla_ops is None:
        _fla_ops = _FLAOps()
    return _fla_ops


class _VarlenTokenShift(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("RWKV7 token shift expects input shape [B, T, D]")
        if x.shape[0] != 1:
            raise ValueError(
                "RWKV7 varlen token shift expects batch size 1 after flattening"
            )

        starts = cu_seqlens[:-1].to(device=x.device, dtype=torch.long)
        ctx.save_for_backward(starts)

        y = torch.empty_like(x)
        if x.shape[1] == 0:
            return y

        y[:, 0, :].copy_(x[:, 0, :]).neg_()
        if x.shape[1] > 1:
            y[:, 1:, :].copy_(x[:, :-1, :])
            y[:, 1:, :].sub_(x[:, 1:, :])

        boundary_starts = starts[starts > 0]
        if boundary_starts.numel() > 0:
            y.index_copy_(
                1,
                boundary_starts,
                x.index_select(1, boundary_starts).neg(),
            )
        return y

    @staticmethod
    def backward(
        ctx: Any,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (starts,) = ctx.saved_tensors

        dx = dy.neg()
        if dy.shape[1] > 1:
            dx[:, :-1, :].add_(dy[:, 1:, :])

        boundary_starts = starts[starts > 0]
        if boundary_starts.numel() > 0:
            prev = boundary_starts - 1
            dx.index_add_(1, prev, dy.index_select(1, boundary_starts).neg())
        return dx, None


def _token_shift_varlen_eager(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return _VarlenTokenShift.apply(x, cu_seqlens)


@torch.compiler.disable
def _token_shift_eager(
    x: torch.Tensor,
    *,
    cp_context: Any | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    if cp_context is not None:
        ops = _require_fla_ops()
        return ops.token_shift_cp(
            x,
            cp_context=cp_context,
            cu_seqlens=cp_context.cu_seqlens,
        )
    if cu_seqlens is not None:
        # FLA's varlen token_shift long kernel autotunes a 3D launch from the
        # number of chunks and sequence spans. Packed batches with many short
        # spans can trip invalid launches during activation checkpointing.
        return _token_shift_varlen_eager(x, cu_seqlens)
    ops = _require_fla_ops()
    return ops.token_shift(x, cu_seqlens)


class RWKVLoRA(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        input_dim: int
        output_dim: int
        low_rank_dim: int
        bias: bool = True
        activation: str | None = "tanh"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.low_rank_dim = config.low_rank_dim
        self.bias = config.bias

        gain = (
            math.sqrt(config.low_rank_dim / config.output_dim)
            if config.low_rank_dim > config.output_dim
            else 1.0
        )
        if config.activation is None:
            activation = Identity()
        elif config.activation == "sigmoid":
            activation = Sigmoid()
        elif config.activation == "tanh":
            activation = Tanh()
        else:
            raise ValueError(f"Unsupported RWKV LoRA activation: {config.activation}")

        self.lora = Sequential(
            _linear(
                config.input_dim,
                config.low_rank_dim,
                bias=False,
                param_init={"weight": _zero_},
            ),
            activation,
            _linear(
                config.low_rank_dim,
                config.output_dim,
                bias=config.bias,
                param_init={
                    "weight": partial(_orthogonal_, gain=gain * 0.1),
                    **({"bias": _zero_} if config.bias else {}),
                },
            ),
        )

    def set_bias_value(self, value: torch.Tensor | float) -> None:
        final = self.lora[2]
        if self.bias and isinstance(final, nn.Linear) and final.bias is not None:
            with torch.no_grad():
                if isinstance(value, torch.Tensor):
                    _copy_tensor_(final.bias, value)
                else:
                    nn.init.constant_(final.bias, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class _ZeroGateLogits(Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_full((*x.shape[:-1], self.output_dim), -math.inf)


class RWKV7TimeMix(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        head_dim: int
        layer_idx: int
        num_hidden_layers: int
        value_dim: int | None = None
        a_low_rank_dim: int = 64
        decay_low_rank_dim: int = 64
        gate_low_rank_dim: int = 128
        v_low_rank_dim: int = 32
        norm_eps: float = 1e-5
        chunk_size: int = 64

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.key_dim = config.hidden_size
        self.value_dim = config.value_dim or config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.head_v_dim = self.value_dim // self.num_heads
        self.layer_idx = config.layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.chunk_size = config.chunk_size

        if self.key_dim % self.num_heads != 0:
            raise ValueError("RWKV7 key dimension must be divisible by num_heads")
        if self.value_dim % self.num_heads != 0:
            raise ValueError("RWKV7 value dimension must be divisible by num_heads")

        self.x_r = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = _linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = _linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = _linear(self.hidden_size, self.value_dim, bias=False)
        self.o_proj = _linear(self.value_dim, self.hidden_size, bias=False)

        self.w_lora = RWKVLoRA.Config(
            input_dim=self.hidden_size,
            output_dim=self.key_dim,
            low_rank_dim=config.decay_low_rank_dim,
            activation="tanh",
        ).build()
        if self.layer_idx == 0:
            self.v_lora = _ZeroGateLogits(self.value_dim)
        else:
            self.v_lora = RWKVLoRA.Config(
                input_dim=self.hidden_size,
                output_dim=self.value_dim,
                low_rank_dim=config.v_low_rank_dim,
                activation=None,
            ).build()
        self.a_lora = RWKVLoRA.Config(
            input_dim=self.hidden_size,
            output_dim=self.key_dim,
            low_rank_dim=config.a_low_rank_dim,
            activation=None,
        ).build()
        self.g_lora = RWKVLoRA.Config(
            input_dim=self.hidden_size,
            output_dim=self.value_dim,
            low_rank_dim=config.gate_low_rank_dim,
            activation="sigmoid",
            bias=False,
        ).build()

        self.g_norm = GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            eps=self.head_dim * config.norm_eps,
            affine=True,
        )

    def _init_self_parameters(self) -> None:
        ratio_0_to_1 = (
            self.layer_idx / (self.num_hidden_layers - 1)
            if self.num_hidden_layers > 1
            else 0.0
        )
        ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)

        device = self.x_r.device
        linear = torch.arange(self.hidden_size, device=device, dtype=torch.float32)
        linear = linear / max(1, self.hidden_size - 1) - 0.5
        ddd = (
            torch.arange(self.hidden_size, device=device, dtype=torch.float32)
            / self.hidden_size
        ).view(1, 1, -1)
        zigzag = (
            (torch.arange(self.hidden_size, device=device, dtype=torch.float32) % self.head_dim)
            - ((self.head_dim - 1) / 2)
        ) / ((self.head_dim - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        www = -6 + 6 * (
            torch.arange(self.hidden_size, device=device, dtype=torch.float32)
            / max(1, self.hidden_size - 1)
        ) ** (1 + ratio_0_to_1**0.3)

        with torch.no_grad():
            _copy_tensor_(
                self.x_r,
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),
            )
            _copy_tensor_(
                self.x_w,
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),
            )
            _copy_tensor_(
                self.x_k,
                1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0),
            )
            _copy_tensor_(
                self.x_v,
                1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0),
            )
            _copy_tensor_(
                self.x_a,
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),
            )
            _copy_tensor_(
                self.x_g,
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),
            )

            nn.init.constant_(self.k_a, 1.02)
            nn.init.constant_(self.r_k, -0.04)
            _copy_tensor_(self.k_k, 0.71 - linear * 0.1)
            self.w_lora.set_bias_value(www + 0.5 + zigzag * 2.5)
            self.a_lora.set_bias_value(-0.19 + zigzag * 0.3 + linear * 0.4)
            if self.layer_idx != 0:
                self.v_lora.set_bias_value(0.73 - linear * 0.4)

            if self.g_norm.weight is not None:
                self.g_norm.weight.fill_(
                    ((self.layer_idx + 1) / self.num_hidden_layers) ** 0.7
                )
            if self.g_norm.bias is not None:
                self.g_norm.bias.zero_()

            _orthogonal_(self.r_proj.weight)
            _orthogonal_(self.k_proj.weight, gain=0.1)
            _orthogonal_(self.v_proj.weight)
            self.o_proj.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        *,
        v_first: torch.Tensor,
        cp_context: Any | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ops = _require_fla_ops()
        batch_size, seq_len, _ = x.shape

        delta = _token_shift_eager(
            x,
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
        )
        xr, xw, xk, xv, xa, xg = ops.fused_addcmul_rwkv7(
            x, delta, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g
        )

        r = self.r_proj(xr)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        layer_v = v
        v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = ops.l2_norm(_reshape_heads(k * self.k_k, self.head_dim))
        k = ops.fused_k_rwkv7(k, a, self.k_a)

        r_heads = _reshape_heads(r, self.head_dim)
        w_heads = _reshape_heads(w, self.head_dim)
        k_heads = _reshape_heads(k, self.head_dim)
        a_heads = _reshape_heads(a, self.head_dim)
        v_heads = _reshape_heads(v, self.head_v_dim)

        o, _ = ops.chunk_dplr_delta_rule(
            q=r_heads,
            k=k_heads,
            v=v_heads,
            a=-kk,
            b=kk * a_heads,
            gk=w_heads,
            scale=1.0,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None if cp_context is not None else cu_seqlens,
            safe_gate=True,
            chunk_size=self.chunk_size,
            cp_context=cp_context,
        )

        o = self.g_norm(_merge_heads(o).view(batch_size * seq_len, self.value_dim))
        o = o.view(batch_size, seq_len, self.value_dim)
        o = ops.gate_output_correction(o, r_heads, k_heads, self.r_k, v_heads, g)
        return self.o_proj(o), layer_v


class RWKV7ChannelMix(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int
        layer_idx: int
        num_hidden_layers: int
        hidden_act: str = "sqrelu"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.layer_idx = config.layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        if config.hidden_act != "sqrelu":
            raise ValueError("RWKV7ChannelMix currently supports hidden_act='sqrelu'")

        self.x_k = nn.Parameter(torch.zeros(self.hidden_size))
        self.key = _linear(self.hidden_size, self.intermediate_size, bias=False)
        self.value = _linear(self.intermediate_size, self.hidden_size, bias=False)

    def _init_self_parameters(self) -> None:
        ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)
        ddd = torch.arange(self.hidden_size, device=self.x_k.device, dtype=torch.float32)
        ddd = ddd / self.hidden_size
        with torch.no_grad():
            _copy_tensor_(
                self.x_k,
                1.0 - torch.pow(ddd, ratio_1_to_almost0**4),
            )
            _orthogonal_(self.key.weight)
            self.value.weight.zero_()

    def forward(
        self,
        x: torch.Tensor,
        *,
        cp_context: Any | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = _token_shift_eager(
            x,
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
        )
        return self.value(_sqrelu(self.key(x.addcmul(delta, self.x_k))))


class RWKV7Block(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int
        num_heads: int
        head_dim: int
        layer_idx: int
        num_hidden_layers: int
        value_dim: int | None = None
        a_low_rank_dim: int = 64
        decay_low_rank_dim: int = 64
        gate_low_rank_dim: int = 128
        v_low_rank_dim: int = 32
        norm_eps: float = 1e-5
        norm_bias: bool = True
        hidden_act: str = "sqrelu"
        chunk_size: int = 64

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layer_idx = config.layer_idx
        self.attn_norm = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.attn = RWKV7TimeMix.Config(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            layer_idx=config.layer_idx,
            num_hidden_layers=config.num_hidden_layers,
            value_dim=config.value_dim,
            a_low_rank_dim=config.a_low_rank_dim,
            decay_low_rank_dim=config.decay_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            v_low_rank_dim=config.v_low_rank_dim,
            norm_eps=config.norm_eps,
            chunk_size=config.chunk_size,
        ).build()
        self.ffn_norm = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.ffn = RWKV7ChannelMix.Config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_idx=config.layer_idx,
            num_hidden_layers=config.num_hidden_layers,
            hidden_act=config.hidden_act,
        ).build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        v_first: torch.Tensor,
        cp_context: Any | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, v_first = self.attn(
            self.attn_norm(x),
            v_first=v_first,
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
        )
        x = x + attn_out
        x = x + self.ffn(
            self.ffn_norm(x),
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
        )
        return x, v_first


class RWKV7Backbone(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int = 65536
        hidden_size: int = 1024
        num_hidden_layers: int = 24
        num_heads: int = 16
        head_dim: int = 64
        intermediate_size: int = 4096
        value_dim: list[int] | None = None
        norm_eps: float = 1e-5
        norm_bias: bool = True
        hidden_act: str = "sqrelu"
        a_low_rank_dim: int = 64
        decay_low_rank_dim: int = 64
        gate_low_rank_dim: int = 128
        v_low_rank_dim: int = 32
        chunk_size: int = 64
        embeddings: Embedding.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.embeddings = (
            config.embeddings
            or Embedding.Config(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                param_init={"weight": _embedding_init},
            )
        ).build()
        self.layers = ModuleDict()
        self.pre_norm = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        value_dims = config.value_dim or [config.hidden_size] * config.num_hidden_layers
        for layer_idx in range(config.num_hidden_layers):
            self.layers[str(layer_idx)] = RWKV7Block.Config(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                num_hidden_layers=config.num_hidden_layers,
                value_dim=value_dims[layer_idx],
                a_low_rank_dim=config.a_low_rank_dim,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                norm_bias=config.norm_bias,
                hidden_act=config.hidden_act,
                chunk_size=config.chunk_size,
            ).build()
        self.norm = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.register_load_state_dict_pre_hook(
            self._remap_legacy_layer0_pre_norm_state_dict
        )

    def _remap_legacy_layer0_pre_norm_state_dict(
        self,
        module: Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        for name in ("weight", "bias"):
            new_key = f"{prefix}pre_norm.{name}"
            old_key = f"{prefix}layers.0.pre_norm.{name}"
            if old_key not in state_dict:
                continue
            target = getattr(self.pre_norm, name, None)
            if target is not None and new_key not in state_dict:
                state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]

    def forward_embeddings(
        self,
        hidden_states: torch.Tensor,
        *,
        cp_context: Any | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states)
        v_first = hidden_states.new_zeros(
            *hidden_states.shape[:-1],
            self.layers["0"].attn.value_dim,
        )
        v_first.requires_grad_(
            torch.is_grad_enabled()
            and (
                hidden_states.requires_grad
                or self.layers["0"].attn.v_proj.weight.requires_grad
            )
        )
        for layer_idx, layer in self.layers.items():
            hidden_states, layer_v = layer(
                hidden_states,
                v_first=v_first,
                cp_context=cp_context,
                cu_seqlens=cu_seqlens,
            )
            if layer_idx == "0":
                v_first = layer_v
        return self.norm(hidden_states)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        cp_context: Any | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_embeddings(
            self.embeddings(tokens),
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
        )


class RWKV7ForCausalLM(BaseModel):
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        vocab_size: int = 65536
        hidden_size: int = 1024
        llm: RWKV7Backbone.Config
        lm_head: Linear.Config | None = None
        uses_fla_context_parallel: bool = True

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            parallelism = trainer_config.parallelism
            training = trainer_config.training
            compile_config = getattr(trainer_config, "compile", None)

            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError("RWKV7 v1 does not support tensor parallelism")
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError("RWKV7 v1 does not support pipeline parallelism")

            if parallelism.context_parallel_degree > 1:
                if parallelism.context_parallel_load_balancer is not None:
                    raise ValueError(
                        "RWKV7 CP requires --parallelism.context_parallel_load_balancer None"
                    )
                total_tokens = training.local_batch_size * training.seq_len
                if total_tokens % parallelism.context_parallel_degree != 0:
                    raise ValueError(
                        f"RWKV7 CP requires local_batch_size * seq_len "
                        f"({total_tokens}) to be divisible by context_parallel_degree "
                        f"({parallelism.context_parallel_degree})"
                    )
                if (
                    compile_config is not None
                    and compile_config.enable
                    and "model" in compile_config.components
                ):
                    logger.warning(
                        "RWKV7 CP with torch.compile is experimental and should "
                        "be checked with benchmarks/rwkv7_compile_bench.py before "
                        "large training runs."
                    )

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            return nparams, 6 * nparams

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.llm = config.llm.build()
        self.lm_head = (
            config.lm_head
            or Linear.Config(
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                bias=False,
                param_init=_output_linear_init(config.hidden_size),
            )
        ).build()
        self._cp_group = None

    def set_cp_process_group(self, cp_group) -> None:
        self._cp_group = cp_group

    def _build_cp_context(
        self,
        cu_seqlens_global: torch.Tensor | None,
        cu_seqlens_global_cpu: torch.Tensor | None,
    ) -> Any | None:
        if self._cp_group is None:
            return None
        if cu_seqlens_global is None:
            raise ValueError("RWKV7 CP requires cu_seqlens_global")
        ops = _require_fla_ops()
        return ops.build_cp_context(
            cu_seqlens_global,
            group=self._cp_group,
            cu_seqlens_cpu=cu_seqlens_global_cpu,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        cu_seqlens_global: torch.Tensor | None = None,
        cu_seqlens_global_cpu: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        cp_context = self._build_cp_context(cu_seqlens_global, cu_seqlens_global_cpu)
        cu_seqlens = cu_seqlens_global if cp_context is None and tokens.shape[0] == 1 else None
        hidden_states = self.llm(tokens, cp_context=cp_context, cu_seqlens=cu_seqlens)
        if self._skip_lm_head:
            return hidden_states
        return self.lm_head(hidden_states)


def rwkv7_backbone_config(
    *,
    vocab_size: int = 65536,
    hidden_size: int = 1024,
    num_hidden_layers: int = 24,
    num_heads: int = 16,
    head_dim: int = 64,
    intermediate_size: int = 4096,
    value_dim: list[int] | None = None,
    norm_eps: float = 1e-5,
    norm_bias: bool = True,
    hidden_act: str = "sqrelu",
    a_low_rank_dim: int = 64,
    decay_low_rank_dim: int = 64,
    gate_low_rank_dim: int = 128,
    v_low_rank_dim: int = 32,
    chunk_size: int = 64,
    skip_embedding_init: bool = False,
) -> RWKV7Backbone.Config:
    embedding_init = {"weight": skip_param_init if skip_embedding_init else _embedding_init}
    return RWKV7Backbone.Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        value_dim=value_dim,
        norm_eps=norm_eps,
        norm_bias=norm_bias,
        hidden_act=hidden_act,
        a_low_rank_dim=a_low_rank_dim,
        decay_low_rank_dim=decay_low_rank_dim,
        gate_low_rank_dim=gate_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        chunk_size=chunk_size,
        embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            param_init=embedding_init,
        ),
    )


def rwkv7_causal_lm_config(
    *,
    vocab_size: int = 65536,
    hidden_size: int = 1024,
    skip_embedding_init: bool = False,
    **kwargs,
) -> RWKV7ForCausalLM.Config:
    return RWKV7ForCausalLM.Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        llm=rwkv7_backbone_config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            skip_embedding_init=skip_embedding_init,
            **kwargs,
        ),
        lm_head=Linear.Config(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=False,
            param_init=_output_linear_init(hidden_size),
        ),
    )
