# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.models.common import Linear, RMSNorm
from torchtitan.models.common.vision_features import (
    _find_vision_spans,
    apply_vision_slices,
)
from torchtitan.models.qwen3_vl.vision_encoder import (
    PatchMerger,
    Qwen3VLVisionEncoder,
)
from torchtitan.models.rwkv7.model import (
    _output_linear_init,
    _zero_,
    LayerNorm,
    RWKV7Backbone,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList, Sequential


ReLU = Module.from_nn_module(nn.ReLU)
GELU = Module.from_nn_module(nn.GELU)


NormKind = Literal["layernorm", "rmsnorm"]
FFNKind = Literal["relu", "gelu", "swiglu"]
ProjectorKind = Literal["mlp", "cross_attn"]


_DEFAULT_BUCKET_LADDER: tuple[int, ...] = (
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
)


def _ceil_to_bucket(n: int, ladder: tuple[int, ...]) -> int:
    """Return the smallest ladder entry >= ``n``.

    Raises ``ValueError`` if ``n`` exceeds the largest bucket — callers must
    ensure ladder coverage matches the dataloader bucket size.
    """
    for b in ladder:
        if b >= n:
            return b
    raise ValueError(
        f"Value {n} exceeds the largest FlexAttention bucket {ladder[-1]}; "
        "increase the projector q_buckets/kv_buckets ladder."
    )


_compiled_create_block_mask = create_block_mask

# Dedicated ``torch.compile`` of ``flex_attention`` for the cross-attn
# projector. Kept separate from ``FlexAttention._compiled_flex_attn`` (used
# by the vision encoder and the LLM) so the projector's distinct (Q_LEN,
# KV_LEN) shapes don't poison the shared compile cache. ``dynamic=False``
# requires the caller to pad Q/K/V to fixed bucket sizes
# (``q_buckets`` / ``kv_buckets`` on the projector config) so each invocation
# specialises against a single shape. Autotune is left at default (off) here
# because turning it on broke the FlexAttention Triton subprocess in earlier
# pipeline smokes — eager Triton kernels are still used inside the compile.
_cross_attn_flex = torch.compile(flex_attention, dynamic=False)


def _build_norm(kind: NormKind, dim: int, eps: float) -> Module:
    if kind == "layernorm":
        return LayerNorm(dim, eps=eps)
    if kind == "rmsnorm":
        return RMSNorm.Config(normalized_shape=dim, eps=eps).build()
    raise ValueError(f"Unknown norm kind: {kind!r}; expected layernorm|rmsnorm")


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
            "RWKV-VL backbone_chunk_size must be at least 16; " f"got {chunk_size}"
        )
    if chunk_size & (chunk_size - 1):
        raise ValueError(
            "RWKV-VL backbone_chunk_size must be a power of two; " f"got {chunk_size}"
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


def _projector_linear_config(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> Linear.Config:
    init = {
        "weight": partial(nn.init.trunc_normal_, std=0.02),
        **({"bias": _zero_} if bias else {}),
    }
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=init,
    )


def _projector_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> Linear:
    return _projector_linear_config(
        in_features, out_features, bias=bias
    ).build()


class _SwiGLUFFN(Module):
    """SwiGLU feed-forward used inside the visual projector.

    Uses param names ``w1``/``w2``/``w3`` to match ``common.FeedForward``.
    """

    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int, bias: bool):
        super().__init__()
        self.w1 = _projector_linear(in_dim, hidden_dim, bias=bias)
        self.w2 = _projector_linear(hidden_dim, out_dim, bias=bias)
        self.w3 = _projector_linear(in_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _build_ffn(
    kind: FFNKind,
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    bias: bool = True,
) -> Module:
    """Build the projector's inner feed-forward block.

    For ``relu``/``gelu`` the structure is ``Linear -> activation -> Linear``,
    keeping state_dict keys ``mlp.0.*`` / ``mlp.2.*`` (back-compat with the
    pre-refactor projector when ``kind='relu'``).
    For ``swiglu`` the module has ``w1``/``w2``/``w3`` Linear params.
    """
    if kind == "relu":
        return Sequential(
            _projector_linear(in_dim, hidden_dim, bias=bias),
            ReLU(),
            _projector_linear(hidden_dim, out_dim, bias=bias),
        )
    if kind == "gelu":
        return Sequential(
            _projector_linear(in_dim, hidden_dim, bias=bias),
            GELU(),
            _projector_linear(hidden_dim, out_dim, bias=bias),
        )
    if kind == "swiglu":
        return _SwiGLUFFN(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, bias=bias
        )
    raise ValueError(f"Unknown ffn kind: {kind!r}; expected relu|gelu|swiglu")


class _VisualStreamProjector(Module):
    def __init__(
        self,
        *,
        encoder_dim: int,
        hidden_dim: int,
        project_dim: int,
        norm_eps: float,
        norm: NormKind = "layernorm",
        ffn: FFNKind = "relu",
    ):
        super().__init__()
        self.pre_norm = _build_norm(norm, project_dim, norm_eps)
        self.mlp = _build_ffn(
            ffn,
            in_dim=encoder_dim,
            hidden_dim=hidden_dim,
            out_dim=project_dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x + self.pre_norm(x)


class _VisualStreamCrossAttnProjector(Module):
    """Cross-attention projector for one DeepStack level.

    Forward projects encoder features into ``(k, v)`` tensors of shape
    ``(N_kv, num_heads, head_dim)``. The actual attention runs inside
    ``attend()``, which is invoked by the caller from inside the LLM's
    ``after_layer`` callback (queries come from the per-layer hidden state).

    Queries have **no Linear projection** (per design) — only ``q_norm`` is
    applied to the gathered ``<image_pad>`` hidden states before the
    head-axis reshape. ``project_dim`` must equal ``num_heads * head_dim``.
    """

    def __init__(
        self,
        *,
        encoder_dim: int,
        project_dim: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        norm_eps: float,
        norm: NormKind = "layernorm",
        ffn: FFNKind = "relu",
        kernel_options: dict | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.project_dim = project_dim
        self.kv_norm = _build_norm(norm, encoder_dim, norm_eps)
        self.k_proj = _projector_linear(
            encoder_dim, num_heads * head_dim, bias=False
        )
        self.v_proj = _projector_linear(
            encoder_dim, num_heads * head_dim, bias=False
        )
        self.q_norm = _build_norm(norm, project_dim, norm_eps)
        self.o_proj = _projector_linear(
            num_heads * head_dim, project_dim, bias=False
        )
        self.ffn_norm = _build_norm(norm, project_dim, norm_eps)
        self.ffn = _build_ffn(
            ffn,
            in_dim=project_dim,
            hidden_dim=hidden_dim,
            out_dim=project_dim,
            bias=True,
        )
        self.kernel_options = kernel_options or {}

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder features to ``(k, v)`` tensors.

        Args:
            x: ``(N_kv, encoder_dim)`` flat-concatenated deepstack features
               for this level (post-merge), padded with zero rows if the
               dataloader bucketed the patch count.

        Returns:
            ``(k, v)`` each of shape ``(N_kv, num_heads, head_dim)``.
        """
        x = self.kv_norm(x)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        return k, v

    def attend(
        self,
        q_real: torch.Tensor,
        k_real: torch.Tensor,
        v_real: torch.Tensor,
        *,
        block_mask: BlockMask,
        q_bucket: int,
        kv_bucket: int,
    ) -> torch.Tensor:
        """Run masked cross-attention for one level and return the per-q delta.

        Args:
            q_real: ``(Q_real, project_dim)`` query hidden states gathered
                from ``<image_pad>`` positions (LLM-side, unnormalized).
            k_real: ``(KV_real, num_heads, head_dim)`` keys.
            v_real: ``(KV_real, num_heads, head_dim)`` values.
            block_mask: precomputed FlexAttention mask of shape
                ``(B=1, H=None, q_bucket, kv_bucket)`` whose ``mask_mod``
                enforces same-image attention and excludes padding rows.
            q_bucket / kv_bucket: padded lengths used to build ``block_mask``.

        Returns:
            ``(Q_real, project_dim)`` post-block update to ADD into the LLM
            hidden state at the ``<image_pad>`` positions that produced
            ``q_real`` (residual already applied internally).
        """
        q_real_len = q_real.shape[0]
        kv_real_len = k_real.shape[0]
        if q_real_len > q_bucket:
            raise ValueError(
                f"q_real_len={q_real_len} exceeds q_bucket={q_bucket}"
            )
        if kv_real_len > kv_bucket:
            raise ValueError(
                f"kv_real_len={kv_real_len} exceeds kv_bucket={kv_bucket}"
            )

        q_in = self.q_norm(q_real)
        # Pad to bucket lengths along the sequence axis.
        q_pad = F.pad(q_in, (0, 0, 0, q_bucket - q_real_len))
        k_pad = F.pad(k_real, (0, 0, 0, 0, 0, kv_bucket - kv_real_len))
        v_pad = F.pad(v_real, (0, 0, 0, 0, 0, kv_bucket - kv_real_len))

        # flex_attention expects ``(bs, heads, seq, dim)``.
        q4 = q_pad.reshape(1, q_bucket, self.num_heads, self.head_dim).transpose(1, 2)
        k4 = k_pad.reshape(1, kv_bucket, self.num_heads, self.head_dim).transpose(1, 2)
        v4 = v_pad.reshape(1, kv_bucket, self.num_heads, self.head_dim).transpose(1, 2)

        attn = _cross_attn_flex(
            q4,
            k4,
            v4,
            block_mask=block_mask,
            kernel_options=self.kernel_options,
        )
        # ``attn`` is ``(1, heads, q_bucket, dim)``. Transpose, trim, flatten.
        attn = attn.transpose(1, 2)[0, :q_real_len].reshape(q_real_len, -1)
        attn = self.o_proj(attn)

        post_attn = q_real + attn
        post_ffn = post_attn + self.ffn(self.ffn_norm(post_attn))
        return post_ffn - q_real  # delta = (q' - q), caller adds to hidden state


class VisualAdapter(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        encoder_dim: int = 1024
        hidden_dim: int | None = None
        project_dim: int = 1024
        num_deepstack: int = 0
        norm_eps: float = 1e-5
        # Selectors shared by both projector kinds. Defaults preserve the
        # original `relu`/`layernorm` projector for back-compat.
        kind: ProjectorKind = "mlp"
        norm: NormKind = "layernorm"
        ffn: FFNKind = "relu"
        # Extra `PatchMerger` applied to the main vision stream when the
        # processor uses a coarser merge size than the vision encoder
        # (image_pad count < K/V token count). The ratio is
        # ``processor_spatial_merge_size / vision_spatial_merge_size`` and
        # must be a positive integer. ``1`` (default) means the extra
        # merger is omitted entirely. Only meaningful when
        # ``kind == "cross_attn"`` — with ``kind == "mlp"`` the deepstack
        # streams also need to match image_pad length, so we currently
        # require the two merge sizes to be equal in that case.
        extra_merge_size: int = 1
        # `cross_attn`-only fields. Unused (and required to be left as default)
        # when ``kind == "mlp"``.
        num_heads: int | None = None
        head_dim: int | None = None
        # Powers-of-two ladders for FlexAttention Q_LEN / KV_LEN buckets. Each
        # forward picks the smallest bucket that fits the current shape and
        # pads the rest with masked rows. ``None`` -> no bucketing (rebuild
        # block_mask per shape).
        q_buckets: tuple[int, ...] | None = None
        kv_buckets: tuple[int, ...] | None = None

    def __init__(self, config: Config):
        super().__init__()
        if config.extra_merge_size < 1:
            raise ValueError(
                "VisualAdapter.Config.extra_merge_size must be >= 1; "
                f"got {config.extra_merge_size}"
            )
        if config.extra_merge_size > 1 and config.kind != "cross_attn":
            raise ValueError(
                "extra_merge_size > 1 (processor merge != vision merge) is "
                "only supported with kind='cross_attn'; with kind='mlp' the "
                "deepstack streams must also match image_pad length and that "
                "compression path isn't implemented yet."
            )
        self.config = config
        self.encoder_dim = config.encoder_dim
        self.project_dim = config.project_dim
        self.hidden_dim = config.hidden_dim or config.project_dim * 4
        self.num_deepstack = config.num_deepstack
        self.kind = config.kind
        self.extra_merge_size = config.extra_merge_size
        self.extra_merger = self._build_extra_merger(config)
        self.main = _VisualStreamProjector(
            encoder_dim=config.encoder_dim,
            hidden_dim=self.hidden_dim,
            project_dim=config.project_dim,
            norm_eps=config.norm_eps,
            norm=config.norm,
            ffn=config.ffn,
        )
        self.deepstack = ModuleList(
            [
                self._build_deepstack_projector(config)
                for _ in range(config.num_deepstack)
            ]
        )

    @staticmethod
    def _build_extra_merger(config: "VisualAdapter.Config") -> Module | None:
        """Build the optional ``PatchMerger`` that compresses the main vision
        stream from vision-encoder length down to image_pad length.

        Returns ``None`` when the merge ratio is 1 (no compression needed).
        """
        if config.extra_merge_size <= 1:
            return None
        merge = config.extra_merge_size
        merged_hidden = config.encoder_dim * (merge**2)
        return PatchMerger(
            hidden_size=config.encoder_dim,
            out_hidden_size=config.encoder_dim,
            spatial_merge_size=merge,
            fc1=_projector_linear_config(merged_hidden, merged_hidden, bias=True),
            fc2=_projector_linear_config(merged_hidden, config.encoder_dim, bias=True),
        )

    @staticmethod
    def _build_deepstack_projector(config: "VisualAdapter.Config") -> Module:
        hidden_dim = config.hidden_dim or config.project_dim * 4
        if config.kind == "mlp":
            return _VisualStreamProjector(
                encoder_dim=config.encoder_dim,
                hidden_dim=hidden_dim,
                project_dim=config.project_dim,
                norm_eps=config.norm_eps,
                norm=config.norm,
                ffn=config.ffn,
            )
        if config.kind == "cross_attn":
            if config.num_heads is None:
                raise ValueError(
                    "VisualAdapter.Config.num_heads is required when "
                    "kind='cross_attn'"
                )
            head_dim = config.head_dim or (config.project_dim // config.num_heads)
            if config.num_heads * head_dim != config.project_dim:
                raise ValueError(
                    f"cross_attn projector requires num_heads*head_dim "
                    f"({config.num_heads}*{head_dim}) == project_dim "
                    f"({config.project_dim}); queries have no Linear projection."
                )
            return _VisualStreamCrossAttnProjector(
                encoder_dim=config.encoder_dim,
                project_dim=config.project_dim,
                num_heads=config.num_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                norm_eps=config.norm_eps,
                norm=config.norm,
                ffn=config.ffn,
            )
        raise ValueError(f"Unknown projector kind: {config.kind!r}")

    def forward(
        self,
        x: torch.Tensor,
        deepstack_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Project the main vision stream and prepare each deepstack stream.

        For ``kind='mlp'`` each deepstack entry is the final projected feature
        tensor (added residually into the LLM hidden state).
        For ``kind='cross_attn'`` each entry is a ``(k, v)`` tuple of shape
        ``(N_kv, num_heads, head_dim)`` — the actual attention runs inside
        the LLM ``after_layer`` callback against the per-layer hidden state.
        """
        if deepstack_features is None:
            deepstack_features = []
        if len(deepstack_features) != self.num_deepstack:
            raise ValueError(
                f"Expected {self.num_deepstack} DeepStack feature tensors, "
                f"got {len(deepstack_features)}."
            )
        if self.extra_merger is not None:
            # PatchMerger expects (batch, seq_len, hidden); the flat path
            # passes (total_tokens, hidden). Each image's token count is
            # divisible by extra_merge_size**2 (enforced by the processor
            # merge-size constraint), so a single unsqueeze/squeeze does
            # the right reshape across image boundaries.
            x = self.extra_merger(x.unsqueeze(0)).squeeze(0)
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
        # Spatial merge size used by the processor / dataloader when counting
        # ``<image_pad>`` tokens. Must be a positive integer multiple of
        # ``vision_encoder.spatial_merge_size``. When equal to vision merge
        # (the default), one ``<image_pad>`` token corresponds to exactly one
        # vision feature; when larger, each ``<image_pad>`` token represents
        # ``(processor_merge/vision_merge)**2`` vision features and the
        # projector compresses the main stream with an extra ``PatchMerger``.
        # Decoupled merge sizes are currently only supported with
        # ``proj.kind == "cross_attn"``.
        processor_spatial_merge_size: int = 2
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

            # Optional projector overrides — let the trainer config override
            # the flavor-baked defaults for projector kind / norm / ffn / heads /
            # extra-merge, and override the processor-side spatial merge size.
            proj_overrides: dict[str, Any] = {}
            for src_name, dst_name in (
                ("projector_kind", "kind"),
                ("projector_norm", "norm"),
                ("projector_ffn", "ffn"),
                ("projector_num_heads", "num_heads"),
                ("projector_head_dim", "head_dim"),
                ("projector_extra_merge_size", "extra_merge_size"),
            ):
                value = getattr(trainer_config, src_name, None)
                if value is not None:
                    proj_overrides[dst_name] = value
            q_bucket = getattr(trainer_config, "projector_q_bucket", None)
            if q_bucket is not None:
                proj_overrides["q_buckets"] = (int(q_bucket),)
            kv_bucket = getattr(trainer_config, "projector_kv_bucket", None)
            if kv_bucket is not None:
                proj_overrides["kv_buckets"] = (int(kv_bucket),)
            if proj_overrides:
                self.proj = replace(self.proj, **proj_overrides)
            processor_merge_override = getattr(
                trainer_config, "processor_spatial_merge_size", None
            )
            if processor_merge_override is not None:
                self.processor_spatial_merge_size = int(processor_merge_override)

            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "RWKV-VL v1 does not support tensor parallelism"
                )
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "RWKV-VL v1 does not support pipeline parallelism"
                )
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
        vision_merge = config.vision_encoder.spatial_merge_size
        processor_merge = config.processor_spatial_merge_size
        if processor_merge < vision_merge or processor_merge % vision_merge != 0:
            raise ValueError(
                "processor_spatial_merge_size must be a positive integer "
                f"multiple of vision_encoder.spatial_merge_size; got "
                f"processor={processor_merge}, vision={vision_merge}"
            )
        extra_merge_size = processor_merge // vision_merge
        if config.proj.extra_merge_size != extra_merge_size:
            config = replace(
                config,
                proj=replace(config.proj, extra_merge_size=extra_merge_size),
            )
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.processor_spatial_merge_size = processor_merge
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
        self._vision_patch_sync_group = None
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

    def set_vision_patch_sync_process_group(self, group) -> None:
        self._vision_patch_sync_group = group

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

    def _sync_flat_vision_patch_bucket(
        self,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not dist.is_available() or not dist.is_initialized():
            return pixel_values, grid_thw

        group = self._vision_patch_sync_group
        world_size = (
            dist.get_world_size(group) if group is not None else dist.get_world_size()
        )
        if world_size == 1:
            return pixel_values, grid_thw

        has_local = (
            pixel_values is not None
            and grid_thw is not None
            and pixel_values.dim() == 2
        )
        local_num_patch = int(pixel_values.shape[0]) if has_local else 0

        max_num_patch = torch.tensor(local_num_patch, dtype=torch.long, device=device)
        dist.all_reduce(max_num_patch, op=dist.ReduceOp.MAX, group=group)
        target_num_patch = int(max_num_patch.item())
        if target_num_patch == 0 or target_num_patch == local_num_patch:
            return pixel_values, grid_thw
        if not has_local:
            return pixel_values, grid_thw
        if target_num_patch < local_num_patch:
            raise RuntimeError(
                f"Rank-synchronized ViT patch bucket target {target_num_patch} "
                f"is smaller than local patch count {local_num_patch}."
            )

        pad_len = target_num_patch - local_num_patch
        pixel_values = torch.cat(
            [
                pixel_values,
                pixel_values.new_zeros((pad_len, pixel_values.shape[1])),
            ],
            dim=0,
        )
        return pixel_values, grid_thw

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Any], torch.Tensor, torch.Tensor]:
        """Run the vision encoder + projector and return (
            main, deepstack_per_level, num_tokens_per_item, num_kv_per_item
        ).

        ``num_tokens_per_item`` is the per-image ``<image_pad>`` count (set
        by ``processor_spatial_merge_size``); ``num_kv_per_item`` is the
        per-image deepstack/vision-merged token count (set by
        ``vision_encoder.spatial_merge_size``). The two are equal when the
        processor and vision merge sizes match.
        """
        pixel_values = pixel_values.to(
            self.vision_encoder.patch_embed.proj.weight.dtype
        )
        merged_embeds, deepstack_features = self.vision_encoder(
            pixel_values,
            grid_thw=grid_thw,
        )
        merged_embeds, deepstack_features = self.proj(
            merged_embeds,
            deepstack_features,
        )
        num_kv_per_item = (
            grid_thw.prod(-1) // self.vision_encoder.spatial_merge_unit
        )
        processor_unit = self.processor_spatial_merge_size**2
        num_tokens_per_item = grid_thw.prod(-1) // processor_unit
        return merged_embeds, deepstack_features, num_tokens_per_item, num_kv_per_item

    def _apply_vision_features(
        self,
        target: torch.Tensor,
        *,
        features: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
        reduce: str,
    ) -> torch.Tensor:
        if global_input_ids is None:
            global_input_ids = local_tokens
            shard_start = 0
        else:
            shard_start = int(global_start.item()) if global_start is not None else 0
        flat_target = target.view(-1, target.shape[-1])
        spans = _find_vision_spans(
            global_input_ids, num_tokens_per_item, vision_token_id
        )
        apply_vision_slices(
            flat_target,
            features,
            spans,
            num_tokens_per_item,
            shard_start=shard_start,
            shard_length=local_tokens.numel(),
            reduce=reduce,
            cast_to_target=(reduce == "add"),
        )
        return target

    def _add_zero_grad_edge(
        self,
        inputs_embeds: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Add a 0-valued autograd edge from each tensor into ``inputs_embeds``.

        FSDP collective backward requires every rank that wraps the projector /
        vision encoder to enter backward; a CP rank or no-image batch may own
        no real vision contribution, so we splice in a zero-valued edge so the
        graph still passes through those modules.
        """
        for tensor in tensors:
            inputs_embeds = inputs_embeds + tensor.sum().to(inputs_embeds.dtype) * 0.0
        return inputs_embeds

    def _empty_projector_outputs(
        self,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Any]]:
        empty = inputs_embeds.new_zeros((0, self.proj.encoder_dim))
        empty_deepstack = [
            inputs_embeds.new_zeros((0, self.proj.encoder_dim))
            for _ in range(self.proj.num_deepstack)
        ]
        return self.proj(empty, empty_deepstack)

    def _make_mlp_injector(
        self,
        *,
        deepstack_features: list[Any],
        num_tokens_per_item: torch.Tensor | None,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
    ):
        """Build an ``after_layer`` callback that adds deepstack features into
        ``<image_pad>`` positions (the original additive DeepStack path).
        """

        def inject(idx: int, layer_hidden_states: torch.Tensor) -> torch.Tensor:
            if idx >= len(deepstack_features) or num_tokens_per_item is None:
                return layer_hidden_states
            return self._apply_vision_features(
                layer_hidden_states,
                features=deepstack_features[idx],
                num_tokens_per_item=num_tokens_per_item,
                vision_token_id=vision_token_id,
                global_input_ids=global_input_ids,
                global_start=global_start,
                local_tokens=local_tokens,
                reduce="add",
            )

        return inject

    def _make_cross_attn_injector(
        self,
        *,
        deepstack_features: list[Any],
        num_tokens_per_item: torch.Tensor | None,
        num_kv_per_item: torch.Tensor | None,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
    ):
        """Build an ``after_layer`` callback that runs masked cross-attention
        between local ``<image_pad>`` queries and projected deepstack K/V.

        The block-diagonal mask (each image_pad attends only to its own
        image's K/V) and bucketed lengths are computed **once per forward**
        and reused across all deepstack levels.
        """
        device = local_tokens.device
        empty_callback = self._make_mlp_injector(
            deepstack_features=deepstack_features,
            num_tokens_per_item=num_tokens_per_item,
            vision_token_id=vision_token_id,
            global_input_ids=global_input_ids,
            global_start=global_start,
            local_tokens=local_tokens,
        )
        if (
            num_tokens_per_item is None
            or num_kv_per_item is None
            or len(deepstack_features) == 0
        ):
            return lambda idx, h: h

        if global_input_ids is None:
            global_input_ids_used = local_tokens
            shard_start = 0
        else:
            global_input_ids_used = global_input_ids
            shard_start = (
                int(global_start.item()) if global_start is not None else 0
            )
        shard_length = local_tokens.numel()

        spans = _find_vision_spans(
            global_input_ids_used, num_tokens_per_item, vision_token_id
        )

        # Gather local Q metadata: contiguous slice ranges in the flat local
        # token stream + per-row image_id label.
        local_ranges: list[tuple[int, int, int]] = []  # (local_start, local_end, image_id)
        for span in spans:
            span_end = span.start + span.length
            overlap_start = max(span.start, shard_start)
            overlap_end = min(span_end, shard_start + shard_length)
            if overlap_start >= overlap_end:
                continue
            local_ranges.append(
                (
                    overlap_start - shard_start,
                    overlap_end - shard_start,
                    span.item_idx,
                )
            )

        if not local_ranges:
            # No image_pad falls on this rank's shard — fall back to a no-op
            # but still consume deepstack features for the autograd edge in
            # the CP zero-grad path.
            del empty_callback
            return lambda idx, h: h

        q_real_len = sum(end - start for start, end, _ in local_ranges)
        kv_real_len = int(num_kv_per_item.sum().item())

        q_ladder = self.proj.config.q_buckets or _DEFAULT_BUCKET_LADDER
        kv_ladder = self.proj.config.kv_buckets or _DEFAULT_BUCKET_LADDER
        q_bucket = _ceil_to_bucket(q_real_len, q_ladder)
        kv_bucket = _ceil_to_bucket(kv_real_len, kv_ladder)

        # Per-row image_id labels, padded with -1 for masked rows.
        q_image_id = torch.full((q_bucket,), -1, dtype=torch.int32, device=device)
        cursor = 0
        for start, end, image_id in local_ranges:
            length = end - start
            q_image_id[cursor : cursor + length] = image_id
            cursor += length

        kv_image_id = torch.full(
            (kv_bucket,), -1, dtype=torch.int32, device=device
        )
        kv_cursor = 0
        for i in range(num_kv_per_item.shape[0]):
            length = int(num_kv_per_item[i].item())
            kv_image_id[kv_cursor : kv_cursor + length] = i
            kv_cursor += length

        def mask_mod(b, h, q_idx, kv_idx):
            q_id = q_image_id[q_idx]
            kv_id = kv_image_id[kv_idx]
            valid_q = q_id >= 0
            valid_kv = kv_id >= 0
            same_image = q_id == kv_id
            # padding rows self-attend so every q has at least one valid kv
            padding_self = (~valid_q) & (~valid_kv) & (q_idx == kv_idx)
            return (same_image & valid_q & valid_kv) | padding_self

        block_mask = _compiled_create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_bucket,
            KV_LEN=kv_bucket,
        )

        # Precompute gather/scatter index tensor for the local Q rows.
        q_index = torch.empty(q_real_len, dtype=torch.long, device=device)
        cursor = 0
        for start, end, _ in local_ranges:
            length = end - start
            q_index[cursor : cursor + length] = torch.arange(
                start, end, device=device
            )
            cursor += length

        def inject(idx: int, layer_hidden_states: torch.Tensor) -> torch.Tensor:
            if idx >= len(deepstack_features):
                return layer_hidden_states
            k_real, v_real = deepstack_features[idx]
            # Trim K/V to real length (deepstack tensors may carry padding
            # rows from the dataloader bucket — those would be masked out
            # via kv_image_id, but kv_real_len <= k_real.shape[0] so we cap.
            k_real = k_real[:kv_real_len]
            v_real = v_real[:kv_real_len]

            flat = layer_hidden_states.reshape(-1, layer_hidden_states.shape[-1])
            q_real = flat.index_select(0, q_index)
            delta = self.proj.deepstack[idx].attend(
                q_real,
                k_real,
                v_real,
                block_mask=block_mask,
                q_bucket=q_bucket,
                kv_bucket=kv_bucket,
            )
            # ``index_add`` (non-inplace) returns a fresh tensor; the in-place
            # variant would overwrite ``layer_hidden_states`` while the previous
            # LLM block's autograd graph still references it as a saved input.
            scattered = flat.index_add(0, q_index, delta.to(flat.dtype))
            return scattered.view_as(layer_hidden_states)

        return inject

    def _prepare_inputs_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        fla_cp_global_input_ids: torch.Tensor | None,
        fla_cp_global_start: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        list[Any],
        torch.Tensor | None,
        torch.Tensor | None,
        int,
    ]:
        inputs_embeds = self.llm.embeddings(tokens)
        image_token_id = (
            special_tokens.get("image_id", self.config.image_token_id)
            if special_tokens is not None
            else self.config.image_token_id
        )
        deepstack_features: list[Any] = []
        num_tokens_per_item: torch.Tensor | None = None
        num_kv_per_item: torch.Tensor | None = None
        if pixel_values is not None and grid_thw is not None:
            (
                merged_embeds,
                deepstack_features,
                num_tokens_per_item,
                num_kv_per_item,
            ) = self._get_vision_embeds(
                pixel_values,
                grid_thw=grid_thw,
            )
            inputs_embeds = self._apply_vision_features(
                inputs_embeds,
                features=merged_embeds,
                num_tokens_per_item=num_tokens_per_item,
                vision_token_id=image_token_id,
                global_input_ids=fla_cp_global_input_ids,
                global_start=fla_cp_global_start,
                local_tokens=tokens,
                reduce="set",
            )
            if fla_cp_global_input_ids is not None:
                # CP v1 computes vision redundantly on every rank, but a rank
                # may own no image placeholder tokens after contiguous sharding.
                zero_grad_tensors = [merged_embeds]
                for entry in deepstack_features:
                    if isinstance(entry, torch.Tensor):
                        zero_grad_tensors.append(entry)
                    else:
                        zero_grad_tensors.extend(entry)
                inputs_embeds = self._add_zero_grad_edge(
                    inputs_embeds, *zero_grad_tensors
                )
        elif "proj" in self._trainable_roots:
            empty_merged, empty_deepstack = self._empty_projector_outputs(inputs_embeds)
            zero_grad_tensors = [empty_merged]
            for entry in empty_deepstack:
                if isinstance(entry, torch.Tensor):
                    zero_grad_tensors.append(entry)
                else:
                    zero_grad_tensors.extend(entry)
            inputs_embeds = self._add_zero_grad_edge(
                inputs_embeds, *zero_grad_tensors
            )
        return (
            inputs_embeds,
            deepstack_features,
            num_tokens_per_item,
            num_kv_per_item,
            image_token_id,
        )

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

        pixel_values, grid_thw = self._sync_flat_vision_patch_bucket(
            pixel_values,
            grid_thw,
            device=tokens.device,
        )

        cp_context = self._build_cp_context(cu_seqlens_global, cu_seqlens_global_cpu)
        cu_seqlens = (
            cu_seqlens_global if cp_context is None and tokens.shape[0] == 1 else None
        )
        (
            inputs_embeds,
            deepstack_features,
            num_tokens_per_item,
            num_kv_per_item,
            image_token_id,
        ) = self._prepare_inputs_embeds(
            tokens,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            special_tokens=special_tokens,
            fla_cp_global_input_ids=fla_cp_global_input_ids,
            fla_cp_global_start=fla_cp_global_start,
        )

        if self.proj.kind == "cross_attn":
            inject_deepstack = self._make_cross_attn_injector(
                deepstack_features=deepstack_features,
                num_tokens_per_item=num_tokens_per_item,
                num_kv_per_item=num_kv_per_item,
                vision_token_id=image_token_id,
                global_input_ids=fla_cp_global_input_ids,
                global_start=fla_cp_global_start,
                local_tokens=tokens,
            )
        else:
            inject_deepstack = self._make_mlp_injector(
                deepstack_features=deepstack_features,
                num_tokens_per_item=num_tokens_per_item,
                vision_token_id=image_token_id,
                global_input_ids=fla_cp_global_input_ids,
                global_start=fla_cp_global_start,
                local_tokens=tokens,
            )

        hidden_states = self.llm.forward_embeddings(
            inputs_embeds,
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
            after_layer=inject_deepstack,
        )
        if self._skip_lm_head:
            return hidden_states
        return self.lm_head(hidden_states)
