# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, replace

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer, MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_chat_datasets import MMChatDataLoader
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MultiModalProcessor,
)
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
    Per-root RWKV-VL learning rates. ``None`` uses the default optimizer
    parameter group's learning rate.
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

    projector_num_heads: int | None = None
    """Number of cross-attention heads (cross_attn projector only)."""

    projector_head_dim: int | None = None
    """Cross-attention head dim (cross_attn projector only)."""

    projector_extra_merge_size: int | None = None
    """Extra projector-side ``PatchMerger`` ratio. The processor's spatial
    merge size and the dataloader collator's merge size are both derived from
    this and the vision encoder's ``spatial_merge_size``
    (``processor_merge = vision_merge * projector_extra_merge_size``).
    ``1`` (default) keeps the existing ``mlp``-projector behavior. Only
    meaningful with ``projector_kind='cross_attn'``."""

    projector_q_bucket: int | None = None
    """Static Q_LEN bucket for the cross-attn projector's FlexAttention call.
    When set, Q is always padded to exactly this length, avoiding dynamic-shape
    recompilation. Should be >= the largest expected ``<image_pad>`` count per
    forward."""

    projector_kv_bucket: int | None = None
    """Static KV_LEN bucket for the cross-attn projector's FlexAttention call.
    When set, K/V are always padded to exactly this length. Should be >= the
    largest expected per-batch vision token count."""


def _rwkv_vl_dataloader(dataset: str, **kwargs) -> GrainDataLoader.Config:
    dataset_config: SingleDatasetConfig = MM_DATASETS[dataset]
    processor = dataset_config.processor
    if not isinstance(processor, MultiModalProcessor.Config):
        raise ValueError(f"Multimodal dataset {dataset!r} has no multimodal processor")
    processor = replace(
        processor,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=2097152,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        **kwargs,
    )
    dataset_config = replace(dataset_config, processor=processor)
    return GrainDataLoader.Config(
        dataset=dataset_config,
        collator=MultiModalCollator.Config(
            max_images_per_batch=128,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
        ),
        streaming_shuffle_buffer_size=128,
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
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(l2_wrap_factor=1e-4)
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**_DEBUG_SPECIAL_TOKENS),
        model_spec=model_registry("debugmodel"),
        dataloader=_rwkv_vl_dataloader("cc12m-test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=512,
            max_context_length=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=SelectiveAC.Config(),
    )


def rwkv_vl_debugmodel_chat() -> Trainer.Config:
    return RWKVVLTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(l2_wrap_factor=1e-4)
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(
            trust_remote_code=True,
            chat_template_add_bos=False,
            chat_template_append_eos=False,
        ),
        model_spec=model_registry("debugmodel"),
        dataloader=_rwkv_vl_chat_dataloader(dataset_path="./tests/assets/cc12m_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=512,
            max_context_length=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=SelectiveAC.Config(),
    )


def _rwkv_vl_chat_config(model_flavor: str) -> Trainer.Config:
    return RWKVVLTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(l2_wrap_factor=1e-4)
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(
            trust_remote_code=True,
            chat_template_add_bos=False,
            chat_template_append_eos=False,
        ),
        model_spec=model_registry(model_flavor),
        dataloader=_rwkv_vl_chat_dataloader(dataset_path="./tests/assets/cc12m_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=512,
            max_context_length=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=True),
        activation_checkpoint=SelectiveAC.Config(),
    )


def rwkv_vl_0_4b_v100m_chat() -> Trainer.Config:
    return _rwkv_vl_chat_config("0.4B-v100M")


def rwkv_vl_1_5b_v100m_chat() -> Trainer.Config:
    return _rwkv_vl_chat_config("1.5B-v100M")


def rwkv_vl_1_5b_v400m_chat() -> Trainer.Config:
    return _rwkv_vl_chat_config("1.5B-v400M")
