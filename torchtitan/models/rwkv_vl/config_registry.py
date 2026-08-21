# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.multimodal.mm_chat_datasets import MMChatDataLoader
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.models.rwkv7.tokenizer import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_VISION_END_TOKEN,
    DEFAULT_VISION_START_TOKEN,
)
from torchtitan.trainer import Trainer

from . import model_registry


_DEBUG_SPECIAL_TOKENS = {
    "image_token": DEFAULT_IMAGE_TOKEN,
    "video_token": "<|video_pad|>",
    "vision_start_token": DEFAULT_VISION_START_TOKEN,
    "vision_end_token": DEFAULT_VISION_END_TOKEN,
    "pad_token": DEFAULT_PAD_TOKEN,
}


@dataclass(kw_only=True, slots=True)
class RWKVVLModuleLRs:
    """
    Per-root RWKV-VL learning rates. ``None`` means use ``optimizer.lr``.
    A value of 0 freezes that root and excludes it from FSDP sharding.
    ``lm_head`` is not configurable; it always follows the resolved ``llm`` LR.
    """

    vision_encoder: float | None = None
    proj: float | None = None
    llm: float | None = None


@dataclass(kw_only=True, slots=True)
class RWKVVLTrainerConfig(Trainer.Config):
    module_lrs: RWKVVLModuleLRs = field(default_factory=RWKVVLModuleLRs)
    """
    Per-root RWKV-VL learning rates. Roots with lr=0 are frozen before the
    optimizer is built and are skipped by selective FSDP sharding.
    """

    backbone_chunk_size: int = 64
    """
    Chunk size used by the RWKV7 backbone DPLR kernels. This does not affect
    state dict shapes; it is applied to the model config before construction.
    """

    projector_kind: str | None = None
    """Override ``proj.kind`` (``mlp`` or ``cross_attn``)."""

    projector_norm: str | None = None
    """Override ``proj.norm`` (``layernorm`` or ``rmsnorm``)."""

    projector_ffn: str | None = None
    """Override ``proj.ffn`` (``relu``, ``gelu``, or ``swiglu``)."""

    projector_num_query_heads: int | None = None
    """Number of GQA query heads (cross_attn projector only)."""

    projector_num_key_value_heads: int | None = None
    """Number of distinct GQA key/value heads (cross_attn only). Each KV head
    serves a group of query heads and owns independently learned channels."""

    tie_projector_qkvo: bool = True
    """Share RWKV Q and visual K/V/output projections across retrieval depths.
    The TokenPacker seed query projection remains separate because its input
    width differs; when untied, its K/V/output projections are separate too."""

    projector_visual_layer_indices: list[int] | None = None
    """Post-block ViT layers exposed as raw DeepStack memories."""

    projector_language_layer_indices: list[int] | None = None
    """RWKV layers after which the corresponding visual memory is retrieved."""

    projector_extra_merge_size: int | None = None
    """TokenPacker query-grid downsampling ratio. The processor's image-token
    merge size is derived from this and the vision encoder's native
    ``spatial_merge_size``
    (``processor_merge = vision_merge * projector_extra_merge_size``).
    ViT patch ordering and collator padding remain at the native merge size.
    ``1`` keeps the native image-token resolution. Only
    meaningful with ``projector_kind='cross_attn'``."""


def _rwkv_vl_dataloader(dataset: str, **kwargs) -> MMDataLoader.Config:
    return MMDataLoader.Config(
        dataset=dataset,
        max_images_per_batch=0,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=2097152,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    )


def _rwkv_vl_chat_dataloader(**kwargs) -> MMChatDataLoader.Config:
    return MMChatDataLoader.Config(
        max_images_per_batch=0,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=2097152,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    )


def rwkv_vl_debugmodel() -> Trainer.Config:
    return RWKVVLTrainerConfig(
        loss=ChunkedCELoss.Config(l2_wrap_factor=1e-4),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(**_DEBUG_SPECIAL_TOKENS),
        model_spec=model_registry("debugmodel"),
        dataloader=_rwkv_vl_dataloader("cc12m-test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def rwkv_vl_debugmodel_chat() -> Trainer.Config:
    return RWKVVLTrainerConfig(
        loss=ChunkedCELoss.Config(l2_wrap_factor=1e-4),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(
            trust_remote_code=True,
            chat_template_add_bos=False,
            chat_template_append_eos=False,
        ),
        model_spec=model_registry("debugmodel"),
        dataloader=_rwkv_vl_chat_dataloader(dataset_path="./tests/assets/cc12m_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def _rwkv_vl_chat_config(model_flavor: str) -> Trainer.Config:
    return RWKVVLTrainerConfig(
        loss=ChunkedCELoss.Config(l2_wrap_factor=1e-4),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(
            trust_remote_code=True,
            chat_template_add_bos=False,
            chat_template_append_eos=False,
        ),
        model_spec=model_registry(model_flavor),
        dataloader=_rwkv_vl_chat_dataloader(dataset_path="./tests/assets/cc12m_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=True),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


def rwkv_vl_0_4b_v100m_chat() -> Trainer.Config:
    return _rwkv_vl_chat_config("0.4B-v100M")


def rwkv_vl_1_5b_v100m_chat() -> Trainer.Config:
    return _rwkv_vl_chat_config("1.5B-v100M")


def rwkv_vl_1_5b_v400m_chat() -> Trainer.Config:
    return _rwkv_vl_chat_config("1.5B-v400M")
