# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import MSELoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.components.validate import Validator
from torchtitan.config import (
    CommConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.protocols.model_spec import ModelSpec
from .data import RAEQwenCollator, RAEQwenProcessor
from .decoder import RAEDecoder
from .discriminator import RAEFeatureDiscriminator
from .encoder import RAEEncoderConfig
from .parallelize import parallelize_rae
from .training import RAEGANAugmentConfig, RAEGANConfig, RAEStage1Trainer


_OPENIMAGES_ROOT = "/mnt/nas/OpenImages/media"
_OPENIMAGES_TRAIN_FILES = "train_*/*.jpg"
_OPENIMAGES_VALIDATION_FILES = "validation*/*.jpg"
_OPENIMAGES_LOCAL_ROOT = "/home/rwkv/molin/openimages_local/data"
_OPENIMAGES_LOCAL_TRAIN_FILES = "train_0/*.jpg"
_OPENIMAGES_LOCAL_VALIDATION_FILES = "validation/*.jpg"
_STATIC_QWEN_MIN_PIXELS = 256 * 256
_STATIC_QWEN_MAX_PIXELS = 1024 * 1024
_STATIC_QWEN_MAX_TOKENS_PER_ITEM = 1024
_STATIC_SEQUENCE_LENGTH = 65536


def model_registry(
    flavor: str = "base",
    *,
    attention_backend: Literal["sdpa", "varlen"] = "sdpa",
    latent_dim: int | None = None,
    decoder_image_size: int | None = None,
    residual_dropout: float = 0.1,
    static_sequence_length: int = 0,
    use_dmuon: bool | None = None,
) -> ModelSpec:
    if flavor not in {"base", "debug"}:
        raise ValueError(f"Unknown RAE flavor: {flavor}")
    debug = flavor == "debug"
    model = RAEDecoder.Config(
        latent_dim=(32 if debug else 1024) if latent_dim is None else latent_dim,
        image_size=(32 if debug else -1)
        if decoder_image_size is None
        else decoder_image_size,
        patch_size=8 if debug else 16,
        hidden_size=64 if debug else 1024,
        num_layers=2 if debug else 8,
        num_heads=4 if debug else 16,
        num_kv_heads=2 if debug else 4,
        intermediate_size=128 if debug else 3072,
        attention_backend=attention_backend,
        spatial_merge_size=2,
        temporal_patch_size=2,
        residual_dropout=residual_dropout,
        static_sequence_length=static_sequence_length,
        use_dmuon=debug if use_dmuon is None else use_dmuon,
    )
    return ModelSpec(
        name="rae",
        flavor=flavor,
        model=model,
        parallelize_fn=parallelize_rae,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
        materialize_before_parallelize=True,
    )


def _image_dataloader(
    *,
    batch_size: int | None = 1,
    dataset_path: str = "tests/assets/cc12m_test",
    data_files: str = "*.tar",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    processor_image_size: int | None = None,
    token_budget: int | None = None,
    max_tokens_per_item: int | None = None,
    image_key: str = "jpg",
    num_prefetch_batches: int = 2,
) -> GrainDataLoader.Config:
    dataset = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path=dataset_path,
            split="train",
            load_dataset_kwargs={"data_files": {"train": data_files}},
        ),
        processor=RAEQwenProcessor.Config(
            model_name="~/models/Qwen3.5-0.8B",
            image_key=image_key,
            image_size=processor_image_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ),
    )
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=RAEQwenCollator.Config(
            batch_size=batch_size,
            media_kind="image",
            token_budget=token_budget,
            max_tokens_per_item=max_tokens_per_item,
        ),
        repeat=True,
        shuffle=True,
        num_prefetch_batches=num_prefetch_batches,
    )


def _dmuon(lr: float) -> OptimizersContainer.Config:
    return OptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="DMuon",
                optimizer_kwargs={
                    "lr": lr,
                    "momentum": 0.95,
                    "weight_decay": 0.0,
                    "adamw_lr": lr,
                    "adamw_weight_decay": 0.0,
                },
            )
        ],
    )


def rae_stage1_debug() -> RAEStage1Trainer.Config:
    encoder = RAEEncoderConfig(
        kind="qwen",
        name="~/models/Qwen3.5-0.8B",
        latent_dim=1024,
        image_size=-1,
        merge_size=2,
        # The debug recipe trains in fp32; flash attention requires half precision.
        attn_implementation="sdpa",
    )
    batch_size = 2
    config = RAEStage1Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry(
            "debug",
            attention_backend="varlen",
            latent_dim=encoder.latent_dim,
            decoder_image_size=-1,
        ),
        loss=MSELoss.Config(),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_wandb=True,
            enable_swanlab=True,
            enable_nvml_metrics=True,
        ),
        dataloader=_image_dataloader(
            batch_size=batch_size,
        ),
        optimizer=_dmuon(2e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=1),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=batch_size,
            max_context_length=1,
            max_norm=1.0,
            steps=10,
            dtype="float32",
            mixed_precision_param="float32",
            disable_cuda_graphs=True,
        ),
        validator=Validator.Config(
            enable=False,
            steps=1,
            dataloader=_image_dataloader(batch_size=batch_size),
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            data_parallel_replicate_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        encoder=encoder,
        gan=RAEGANConfig(
            discriminator_start_step=0,
            discriminator_update_start_step=0,
            perceptual_start_step=0,
        ),
    )
    return config


def rae_stage1_dmuon() -> RAEStage1Trainer.Config:
    config = rae_stage1_debug()
    encoder = RAEEncoderConfig(
        kind="qwen",
        name="~/models/Qwen3.5-0.8B",
        latent_dim=1024,
        image_size=-1,
        layer_indices=(2, 5, 8, 11),
        merge_size=2,
        dtype="bfloat16",
        # torch.compile re-specializes on each batch's packed token count and
        # grid-row count through HF's data-dependent graph breaks, causing a
        # recompile storm across ranks; bf16 + flash varlen already gives
        # ~7x over the fp32 sdpa default. Re-enable encoder.compile only with
        # a static input shape.
    )
    config.encoder = encoder
    config.model_spec = model_registry(
        "base",
        latent_dim=encoder.latent_dim,
        decoder_image_size=-1,
        use_dmuon=True,
    )
    # Post-merger documents are capped at 1024 tokens by max_pixels, which
    # bounds the attention context for the decoder FLOPs estimate.
    config.model_spec.model.flops_attention_context = 1024
    config.dataloader = _image_dataloader(
        batch_size=1,
    )
    config.training = TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=1,
        max_context_length=1,
        max_norm=1.0,
        steps=10000,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        disable_cuda_graphs=True,
    )
    config.optimizer = _dmuon(2e-4)
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=625,
        decay_type="cosine",
        min_lr_factor=0.1,
    )
    config.gan = RAEGANConfig(
        ema_decay=0.9978,
        perceptual_kind="lpips",
        lpips_calibration_checkpoint_path="pretrained_models/lpips/vgg_lpips.pth",
        lpips_vgg_checkpoint_path="pretrained_models/lpips/vgg16-397923af.pth",
        augment=RAEGANAugmentConfig(probability=1.0, cutout=0.0),
        discriminator_warmup_steps=625,
    )
    config.discriminator = RAEFeatureDiscriminator.Config(
        feature_channels=768,
        backbone_kind="hf",
        hf_model_path="~/models/dinov3-vitb16-pretrain-lvd1689m",
        hf_key_depths=(2, 5, 8, 11),
        backbone_batch_size=64,
    )
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=4,
        data_parallel_shard_degree=1,
    )
    config.checkpoint = CheckpointManager.Config(enable=True, interval=1000)
    return config


def rae_stage1_openimages() -> RAEStage1Trainer.Config:
    """Full RAEv2 recipe streaming OpenImages train and validation folders."""
    config = rae_stage1_dmuon()
    config.dataloader = _image_dataloader(
        batch_size=1,
        dataset_path=_OPENIMAGES_ROOT,
        data_files=_OPENIMAGES_TRAIN_FILES,
        image_key="image",
    )
    config.validator.dataloader = _image_dataloader(
        batch_size=1,
        dataset_path=_OPENIMAGES_ROOT,
        data_files=_OPENIMAGES_VALIDATION_FILES,
        image_key="image",
    )
    config.validator.enable = True
    config.validator.steps = -1
    config.validator.freq = 1000
    return config


def _openimages_static(static_sequence_length: int) -> RAEStage1Trainer.Config:
    """Locally staged OpenImages recipe with static token packing and graphs."""
    config = _dmuon_static(static_sequence_length)
    token_budget = static_sequence_length - _STATIC_QWEN_MAX_TOKENS_PER_ITEM
    config.dataloader = _image_dataloader(
        batch_size=None,
        dataset_path=_OPENIMAGES_LOCAL_ROOT,
        data_files=_OPENIMAGES_LOCAL_TRAIN_FILES,
        image_key="image",
        min_pixels=_STATIC_QWEN_MIN_PIXELS,
        max_pixels=_STATIC_QWEN_MAX_PIXELS,
        token_budget=token_budget,
        max_tokens_per_item=_STATIC_QWEN_MAX_TOKENS_PER_ITEM,
        num_prefetch_batches=4,
    )
    config.validator.dataloader = _image_dataloader(
        batch_size=None,
        dataset_path=_OPENIMAGES_LOCAL_ROOT,
        data_files=_OPENIMAGES_LOCAL_VALIDATION_FILES,
        image_key="image",
        min_pixels=_STATIC_QWEN_MIN_PIXELS,
        max_pixels=_STATIC_QWEN_MAX_PIXELS,
        token_budget=token_budget,
        max_tokens_per_item=_STATIC_QWEN_MAX_TOKENS_PER_ITEM,
        num_prefetch_batches=4,
    )
    config.validator.enable = True
    config.validator.steps = -1
    config.validator.freq = 1000
    return config


def rae_stage1_openimages_static() -> RAEStage1Trainer.Config:
    return _openimages_static(_STATIC_SEQUENCE_LENGTH)


def rae_stage1_openimages_static_128k() -> RAEStage1Trainer.Config:
    """Doubled 131072-token static capacity for high-utilization runs."""
    return _openimages_static(2 * _STATIC_SEQUENCE_LENGTH)


def rae_stage1_openimages_static_96k() -> RAEStage1Trainer.Config:
    """98304-token static capacity with more allocator headroom than 128k."""
    return _openimages_static(3 * _STATIC_SEQUENCE_LENGTH // 2)


def _dmuon_static(static_sequence_length: int) -> RAEStage1Trainer.Config:
    """Throughput recipe with a fixed packed-token budget.

    Qwen keeps each image's aspect ratio while constraining its pixel area to
    at most 1024x1024. The post-merge token ceiling is 1024 per row; the
    collator fills the budget (static capacity minus one max-size item for the
    isolated padding document) with a variable number of rows.
    """
    config = rae_stage1_dmuon()
    token_budget = static_sequence_length - _STATIC_QWEN_MAX_TOKENS_PER_ITEM
    config.model_spec = model_registry(
        "base",
        attention_backend="varlen",
        latent_dim=config.encoder.latent_dim,
        decoder_image_size=-1,
        static_sequence_length=static_sequence_length,
        use_dmuon=True,
    )
    config.model_spec.model.flops_attention_context = _STATIC_QWEN_MAX_TOKENS_PER_ITEM
    config.training = TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=token_budget,
        num_tokens_per_train_step=token_budget * 4,
        max_context_length=1,
        max_norm=1.0,
        steps=10000,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        disable_cuda_graphs=False,
    )
    config.dataloader = _image_dataloader(
        batch_size=None,
        min_pixels=_STATIC_QWEN_MIN_PIXELS,
        max_pixels=_STATIC_QWEN_MAX_PIXELS,
        token_budget=token_budget,
        max_tokens_per_item=_STATIC_QWEN_MAX_TOKENS_PER_ITEM,
        num_prefetch_batches=4,
    )
    config.validator.dataloader = _image_dataloader(
        batch_size=None,
        min_pixels=_STATIC_QWEN_MIN_PIXELS,
        max_pixels=_STATIC_QWEN_MAX_PIXELS,
        token_budget=token_budget,
        max_tokens_per_item=_STATIC_QWEN_MAX_TOKENS_PER_ITEM,
        num_prefetch_batches=4,
    )
    config.activation_checkpoint = FullAC.Config()
    config.compile = CompileConfig(enable=True, components=["model", "discriminator"])
    # Four ranks compile the encoder/decoder/discriminator concurrently on the
    # first step, and mid-run CUDA-graph captures pause collectives; loosen the
    # NCCL watchdog bounds accordingly.
    config.comm = CommConfig(init_timeout_seconds=3600, train_timeout_seconds=600)
    return config


def rae_stage1_dmuon_static() -> RAEStage1Trainer.Config:
    return _dmuon_static(_STATIC_SEQUENCE_LENGTH)


def rae_stage1_dmuon_static_128k() -> RAEStage1Trainer.Config:
    """Doubled 131072-token static capacity for high-utilization runs."""
    return _dmuon_static(2 * _STATIC_SEQUENCE_LENGTH)


__all__ = [
    "model_registry",
    "rae_stage1_debug",
    "rae_stage1_dmuon",
    "rae_stage1_dmuon_static",
    "rae_stage1_dmuon_static_128k",
    "rae_stage1_openimages",
    "rae_stage1_openimages_static",
    "rae_stage1_openimages_static_128k",
    "rae_stage1_openimages_static_96k",
]
