# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.models.common import Linear
from torchtitan.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder
from torchtitan.models.rwkv7.model import (
    LayerNorm,
    RWKV7Backbone,
    _output_linear_init,
    _zero_,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList, Sequential


ReLU = Module.from_nn_module(nn.ReLU)


_ROOT_MODULE_NAMES = ("vision_encoder", "proj", "llm", "lm_head")
_ROOT_PARAM_PATTERNS = {
    "vision_encoder": r"^vision_encoder\.",
    "proj": r"^proj\.",
    "llm": r"^llm\.",
    "lm_head": r"^lm_head\.",
}


def _default_root_lrs() -> dict[str, float]:
    return {name: 1.0 for name in _ROOT_MODULE_NAMES}


def _validate_root_lrs(root_lrs: dict[str, float]) -> dict[str, float]:
    missing = set(_ROOT_MODULE_NAMES) - set(root_lrs)
    unknown = set(root_lrs) - set(_ROOT_MODULE_NAMES)
    if missing or unknown:
        raise ValueError(
            "RWKV-VL root_lrs must contain exactly "
            f"{list(_ROOT_MODULE_NAMES)}; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    resolved = {name: float(root_lrs[name]) for name in _ROOT_MODULE_NAMES}
    negative = {name: lr for name, lr in resolved.items() if lr < 0}
    if negative:
        raise ValueError(f"RWKV-VL module LRs must be non-negative, got {negative}")
    if not any(lr > 0 for lr in resolved.values()):
        raise ValueError("At least one RWKV-VL module LR must be greater than 0")
    return resolved


def _resolve_root_lrs(module_lrs: Any, default_lr: float) -> dict[str, float]:
    if default_lr <= 0:
        raise ValueError(
            "RWKV-VL module LR config requires --optimizer.lr to be greater than 0"
        )

    resolved = {}
    for name in _ROOT_MODULE_NAMES:
        value = getattr(module_lrs, name)
        resolved[name] = default_lr if value is None else float(value)

    if getattr(module_lrs, "lm_head") is None:
        resolved["lm_head"] = resolved["llm"]

    return _validate_root_lrs(resolved)


def _configure_optimizer_param_groups(
    optimizer_config: Any,
    root_lrs: dict[str, float],
):
    base_lr = float(optimizer_config.lr)
    if base_lr <= 0:
        raise ValueError(
            "RWKV-VL module LR config requires --optimizer.lr to be greater than 0"
        )

    optimizer_config.param_groups = [
        ParamGroupConfig(
            pattern=_ROOT_PARAM_PATTERNS[name],
            lr_multiplier=lr / base_lr,
        )
        for name, lr in root_lrs.items()
        if lr > 0
    ]


def _validate_backbone_chunk_size(chunk_size: int) -> int:
    chunk_size = int(chunk_size)
    if chunk_size < 16:
        raise ValueError(
            "RWKV-VL backbone_chunk_size must be at least 16; "
            f"got {chunk_size}"
        )
    if chunk_size & (chunk_size - 1):
        raise ValueError(
            "RWKV-VL backbone_chunk_size must be a power of two; "
            f"got {chunk_size}"
        )
    return chunk_size


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> Linear:
    init = {"weight": _zero_, "bias": _zero_} if bias else {"weight": _zero_}
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=init,
    ).build()


def _projector_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> Linear:
    init = {
        "weight": partial(nn.init.trunc_normal_, std=0.02),
        **({"bias": _zero_} if bias else {}),
    }
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=init,
    ).build()


class _VisualStreamProjector(Module):
    def __init__(
        self,
        *,
        encoder_dim: int,
        hidden_dim: int,
        project_dim: int,
        norm_eps: float,
    ):
        super().__init__()
        self.pre_norm = LayerNorm(project_dim, eps=norm_eps)
        self.mlp = Sequential(
            _projector_linear(encoder_dim, hidden_dim, bias=True),
            ReLU(),
            _projector_linear(hidden_dim, project_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x + self.pre_norm(x)


class VisualAdapter(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        encoder_dim: int = 1024
        hidden_dim: int | None = None
        project_dim: int = 1024
        num_deepstack: int = 0
        norm_eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder_dim = config.encoder_dim
        self.project_dim = config.project_dim
        self.hidden_dim = config.hidden_dim or config.project_dim * 4
        self.num_deepstack = config.num_deepstack
        self.main = _VisualStreamProjector(
            encoder_dim=config.encoder_dim,
            hidden_dim=self.hidden_dim,
            project_dim=config.project_dim,
            norm_eps=config.norm_eps,
        )
        self.deepstack = ModuleList(
            [
                _VisualStreamProjector(
                    encoder_dim=config.encoder_dim,
                    hidden_dim=self.hidden_dim,
                    project_dim=config.project_dim,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.num_deepstack)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        deepstack_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if deepstack_features is None:
            deepstack_features = []
        if len(deepstack_features) != self.num_deepstack:
            raise ValueError(
                f"Expected {self.num_deepstack} DeepStack feature tensors, "
                f"got {len(deepstack_features)}."
            )
        projected_deepstack = [
            projector(feature)
            for projector, feature in zip(self.deepstack, deepstack_features)
        ]
        return self.main(x), projected_deepstack


class RWKV7VLForConditionalGeneration(BaseModel):
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        vocab_size: int = 65536
        hidden_size: int = 1024
        llm: RWKV7Backbone.Config
        vision_encoder: Qwen3VLVisionEncoder.Config
        proj: VisualAdapter.Config
        lm_head: Linear.Config | None = None
        image_token_id: int = 65532
        vision_start_token_id: int = 65530
        vision_end_token_id: int = 65531
        uses_fla_context_parallel: bool = True
        root_lrs: dict[str, float] = field(default_factory=_default_root_lrs)

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            parallelism = trainer_config.parallelism
            training = trainer_config.training
            compile_config = getattr(trainer_config, "compile", None)
            module_lrs = getattr(trainer_config, "module_lrs")
            self.root_lrs = _resolve_root_lrs(module_lrs, trainer_config.optimizer.lr)
            _configure_optimizer_param_groups(
                trainer_config.optimizer,
                self.root_lrs,
            )
            self.llm = replace(
                self.llm,
                chunk_size=_validate_backbone_chunk_size(
                    getattr(trainer_config, "backbone_chunk_size", self.llm.chunk_size)
                ),
            )

            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError("RWKV-VL v1 does not support tensor parallelism")
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError("RWKV-VL v1 does not support pipeline parallelism")
            if parallelism.context_parallel_degree > 1:
                if parallelism.context_parallel_load_balancer is not None:
                    raise ValueError(
                        "RWKV-VL CP requires --parallelism.context_parallel_load_balancer None"
                    )
                total_tokens = training.local_batch_size * training.seq_len
                if total_tokens % parallelism.context_parallel_degree != 0:
                    raise ValueError(
                        f"RWKV-VL CP requires local_batch_size * seq_len "
                        f"({total_tokens}) to be divisible by context_parallel_degree "
                        f"({parallelism.context_parallel_degree})"
                    )
                if (
                    compile_config is not None
                    and compile_config.enable
                    and "model" in compile_config.components
                ):
                    from torchtitan.tools.logging import logger

                    logger.warning(
                        "RWKV-VL CP with torch.compile is experimental and should "
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
        self.vision_encoder = config.vision_encoder.build()
        self.proj = config.proj.build()
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
        self._trainable_roots = self._apply_root_lr_selection()

    def _apply_root_lr_selection(self) -> tuple[str, ...]:
        root_lrs = _validate_root_lrs(self.config.root_lrs)
        module_roots = {
            "vision_encoder": self.vision_encoder,
            "proj": self.proj,
            "llm": self.llm,
            "lm_head": self.lm_head,
        }
        for name, module in module_roots.items():
            module.requires_grad_(root_lrs[name] > 0)
        return tuple(name for name in _ROOT_MODULE_NAMES if root_lrs[name] > 0)

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
            raise ValueError("RWKV-VL CP requires cu_seqlens_global")
        from torchtitan.models.rwkv7.model import _require_fla_ops

        ops = _require_fla_ops()
        return ops.build_cp_context(
            cu_seqlens_global,
            group=self._cp_group,
            cu_seqlens_cpu=cu_seqlens_global_cpu,
        )

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.proj.weight.dtype)
        merged_embeds, deepstack_features = self.vision_encoder(
            pixel_values,
            grid_thw=grid_thw,
        )
        merged_embeds, deepstack_features = self.proj(
            merged_embeds,
            deepstack_features,
        )
        num_tokens_per_item = grid_thw.prod(-1) // self.vision_encoder.spatial_merge_unit
        return merged_embeds, deepstack_features, num_tokens_per_item

    def _global_vision_positions(
        self,
        *,
        global_tokens: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
    ) -> list[tuple[int, int, int]]:
        flat = global_tokens.reshape(-1)
        mask = flat == vision_token_id
        prev = torch.cat([torch.zeros(1, dtype=torch.bool, device=mask.device), mask[:-1]])
        starts = torch.where(mask & ~prev)[0]
        positions = []
        for item_idx in range(num_tokens_per_item.shape[0]):
            positions.append(
                (
                    item_idx,
                    int(starts[item_idx].item()),
                    int(num_tokens_per_item[item_idx].item()),
                )
            )
        return positions

    def _scatter_vision_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        merged_embeds: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if global_input_ids is None:
            global_input_ids = local_tokens
            shard_start = 0
        else:
            shard_start = int(global_start.item()) if global_start is not None else 0

        feature_offsets = None
        if merged_embeds.dim() == 2:
            feature_offsets = torch.cat(
                [
                    torch.zeros(
                        1,
                        dtype=torch.long,
                        device=num_tokens_per_item.device,
                    ),
                    num_tokens_per_item.to(torch.long).cumsum(0),
                ]
            )

        shard_end = shard_start + local_tokens.numel()
        for item_idx, start, n_tokens in self._global_vision_positions(
            global_tokens=global_input_ids,
            num_tokens_per_item=num_tokens_per_item,
            vision_token_id=vision_token_id,
        ):
            end = start + n_tokens
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - shard_start
            feature_start = overlap_start - start
            feature_len = overlap_end - overlap_start
            if feature_offsets is None:
                vision_slice = merged_embeds[
                    item_idx, feature_start : feature_start + feature_len
                ]
            else:
                item_offset = int(feature_offsets[item_idx].item())
                vision_slice = merged_embeds[
                    item_offset
                    + feature_start : item_offset
                    + feature_start
                    + feature_len
                ]
            inputs_embeds.view(-1, inputs_embeds.shape[-1])[
                local_start : local_start + feature_len
            ] = vision_slice
        return inputs_embeds

    def _add_vision_embeds(
        self,
        hidden_states: torch.Tensor,
        *,
        vision_embeds: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if global_input_ids is None:
            global_input_ids = local_tokens
            shard_start = 0
        else:
            shard_start = int(global_start.item()) if global_start is not None else 0

        feature_offsets = None
        if vision_embeds.dim() == 2:
            feature_offsets = torch.cat(
                [
                    torch.zeros(
                        1,
                        dtype=torch.long,
                        device=num_tokens_per_item.device,
                    ),
                    num_tokens_per_item.to(torch.long).cumsum(0),
                ]
            )

        shard_end = shard_start + local_tokens.numel()
        for item_idx, start, n_tokens in self._global_vision_positions(
            global_tokens=global_input_ids,
            num_tokens_per_item=num_tokens_per_item,
            vision_token_id=vision_token_id,
        ):
            end = start + n_tokens
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - shard_start
            feature_start = overlap_start - start
            feature_len = overlap_end - overlap_start
            if feature_offsets is None:
                vision_slice = vision_embeds[
                    item_idx, feature_start : feature_start + feature_len
                ]
            else:
                item_offset = int(feature_offsets[item_idx].item())
                vision_slice = vision_embeds[
                    item_offset
                    + feature_start : item_offset
                    + feature_start
                    + feature_len
                ]
            hidden_states.view(-1, hidden_states.shape[-1])[
                local_start : local_start + feature_len
            ] += vision_slice.to(hidden_states.dtype)
        return hidden_states

    def _prepare_inputs_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        fla_cp_global_input_ids: torch.Tensor | None,
        fla_cp_global_start: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None, int]:
        inputs_embeds = self.llm.embeddings(tokens)
        image_token_id = (
            special_tokens.get("image_id", self.config.image_token_id)
            if special_tokens is not None
            else self.config.image_token_id
        )
        deepstack_features: list[torch.Tensor] = []
        num_tokens_per_item = None
        if pixel_values is not None and grid_thw is not None:
            merged_embeds, deepstack_features, num_tokens_per_item = self._get_vision_embeds(
                pixel_values,
                grid_thw=grid_thw,
            )
            inputs_embeds = self._scatter_vision_embeds(
                inputs_embeds,
                merged_embeds=merged_embeds,
                num_tokens_per_item=num_tokens_per_item,
                vision_token_id=image_token_id,
                global_input_ids=fla_cp_global_input_ids,
                global_start=fla_cp_global_start,
                local_tokens=tokens,
            )
            if fla_cp_global_input_ids is not None:
                # CP v1 computes vision redundantly on every rank, but a rank
                # may own no image placeholder tokens after contiguous sharding.
                # Keep a zero-valued autograd edge so FSDP-wrapped vision
                # modules enter backward collectively on all CP ranks.
                inputs_embeds = inputs_embeds + merged_embeds.sum().to(
                    inputs_embeds.dtype
                ) * 0.0
                for deepstack_embeds in deepstack_features:
                    inputs_embeds = inputs_embeds + deepstack_embeds.sum().to(
                        inputs_embeds.dtype
                    ) * 0.0
        return inputs_embeds, deepstack_features, num_tokens_per_item, image_token_id

    def _forward_llm_with_deepstack(
        self,
        hidden_states: torch.Tensor,
        *,
        deepstack_features: list[torch.Tensor],
        num_tokens_per_item: torch.Tensor | None,
        vision_token_id: int,
        tokens: torch.Tensor,
        fla_cp_global_input_ids: torch.Tensor | None,
        fla_cp_global_start: torch.Tensor | None,
        cp_context: Any | None,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        v_first = None
        for layer_idx, layer in self.llm.layers.items():
            hidden_states, v_first = layer(
                hidden_states,
                v_first=v_first,
                cp_context=cp_context,
                cu_seqlens=cu_seqlens,
            )
            idx = int(layer_idx)
            if idx < len(deepstack_features) and num_tokens_per_item is not None:
                hidden_states = self._add_vision_embeds(
                    hidden_states,
                    vision_embeds=deepstack_features[idx],
                    num_tokens_per_item=num_tokens_per_item,
                    vision_token_id=vision_token_id,
                    global_input_ids=fla_cp_global_input_ids,
                    global_start=fla_cp_global_start,
                    local_tokens=tokens,
                )
        return self.llm.norm(hidden_states)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        cu_seqlens_global: torch.Tensor | None = None,
        cu_seqlens_global_cpu: torch.Tensor | None = None,
        fla_cp_global_input_ids: torch.Tensor | None = None,
        fla_cp_global_start: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("RWKV-VL video inputs are not implemented yet")

        cp_context = self._build_cp_context(cu_seqlens_global, cu_seqlens_global_cpu)
        cu_seqlens = cu_seqlens_global if cp_context is None and tokens.shape[0] == 1 else None
        (
            inputs_embeds,
            deepstack_features,
            num_tokens_per_item,
            image_token_id,
        ) = self._prepare_inputs_embeds(
            tokens,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            special_tokens=special_tokens,
            fla_cp_global_input_ids=fla_cp_global_input_ids,
            fla_cp_global_start=fla_cp_global_start,
        )
        hidden_states = self._forward_llm_with_deepstack(
            inputs_embeds,
            deepstack_features=deepstack_features,
            num_tokens_per_item=num_tokens_per_item,
            vision_token_id=image_token_id,
            tokens=tokens,
            fla_cp_global_input_ids=fla_cp_global_input_ids,
            fla_cp_global_start=fla_cp_global_start,
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
        )
        if self._skip_lm_head:
            return hidden_states
        return self.lm_head(hidden_states)
