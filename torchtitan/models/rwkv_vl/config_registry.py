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
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.multimodal.mm_chat_datasets import MMChatDataLoader
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader
from torchtitan.models.rwkv_vl.tokenizer import RwkvVLMultiModalTokenizer
from torchtitan.trainer import Trainer

from . import model_registry


_DEBUG_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


@dataclass(kw_only=True, slots=True)
class RWKVVLModuleLRs:
    """
    Per-root RWKV-VL learning rates. ``None`` means use ``optimizer.lr``.
    A value of 0 freezes that root and excludes it from FSDP sharding.
    ``lm_head`` defaults to the resolved ``llm`` LR when left as ``None``.
    """

    vision_encoder: float | None = None
    proj: float | None = None
    llm: float | None = None
    lm_head: float | None = None


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
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**_DEBUG_SPECIAL_TOKENS),
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
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=RwkvVLMultiModalTokenizer.Config(),
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
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=RwkvVLMultiModalTokenizer.Config(),
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
