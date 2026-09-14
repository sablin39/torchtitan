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
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.protocols.model_spec import ModelSpec
from .data import RAEQwenCollator, RAEQwenProcessor
from .decoder import RAEDecoder
from .discriminator import RAEFeatureDiscriminator
from .encoder import RAEEncoderConfig
from .parallelize import parallelize_rae
from .trainer import RAEGANAugmentConfig, RAEGANConfig, RAEStage1Trainer


# Full OpenImages train set as gzipped webdataset tars (16 shards, members
# named <folder>/<hash>.jpg, row key "jpg"). The single validation.tar.gz has
# one shard, too few to split across DP ranks, so validation reads a 25k-image
# subset extracted from its head into a plain folder instead.
_OPENIMAGES_TAR_ROOT = "/mnt/sda1/OpenImages/tar"
_OPENIMAGES_TAR_TRAIN_FILES = "train_*.tar.gz"
_OPENIMAGES_VALIDATION_SUBSET_ROOT = "/mnt/sda1/OpenImages/validation_subset"
_OPENIMAGES_VALIDATION_SUBSET_FILES = "validation/*.jpg"
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
    long_skip_connections: tuple[tuple[int, int], ...] = (),
    use_dmuon: bool | None = None,
) -> ModelSpec:
    if flavor not in {"base", "debug"}:
        raise ValueError(f"Unknown RAE flavor: {flavor}")
    debug = flavor == "debug"
    sizes = {
        "latent_dim": 32 if debug else 1024,
        "image_size": 32 if debug else -1,
        "patch_size": 8 if debug else 16,
        "hidden_size": 64 if debug else 1024,
        "num_layers": 2 if debug else 8,
        "num_heads": 4 if debug else 16,
        "num_kv_heads": 2 if debug else 4,
        "intermediate_size": 128 if debug else 3072,
    }
    if latent_dim is not None:
        sizes["latent_dim"] = latent_dim
    if decoder_image_size is not None:
        sizes["image_size"] = decoder_image_size
    model = RAEDecoder.Config(
        **sizes,
        attention_backend=attention_backend,
        spatial_merge_size=2,
        temporal_patch_size=2,
        residual_dropout=residual_dropout,
        static_sequence_length=static_sequence_length,
        long_skip_connections=long_skip_connections,
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
    token_budget: int | None = None,
    max_tokens_per_item: int | None = None,
    image_key: str = "jpg",
    num_prefetch_batches: int = 2,
    num_processor_workers: int = 0,
    readahead_mb: int = 0,
) -> GrainDataLoader.Config:
    dataset = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path=dataset_path,
            split="train",
            load_dataset_kwargs={"data_files": {"train": data_files}},
            # Rows cross the process-pool boundary when num_processor_workers
            # > 0; keep the image as encoded bytes so JPEG decode happens in
            # the workers instead of at submit-pickle time.
            decode_images=False,
            readahead_mb=readahead_mb,
        ),
        processor=RAEQwenProcessor.Config(
            model_name="~/models/Qwen3.5-0.8B",
            image_key=image_key,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ),
    )
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=RAEQwenCollator.Config(
            batch_size=batch_size,
            token_budget=token_budget,
            max_tokens_per_item=max_tokens_per_item,
        ),
        repeat=True,
        shuffle=True,
        num_prefetch_batches=num_prefetch_batches,
        num_processor_workers=num_processor_workers,
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
                    # Muon's updates are scale-invariant to the weight norm, so
                    # a small decoupled decay keeps norms (and the effective
                    # LR) from drifting over long runs. Norm-gain/cls params
                    # route to the AdamW subgroup, which stays decay-free.
                    "weight_decay": 0.01,
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
            # Keep the debug recipe on the pre-feature-matching loss.
            feature_matching_weight=0.0,
        ),
    )
    return config


def rae_stage1_openimages_static_96k_uvit() -> RAEStage1Trainer.Config:
    """Production recipe: 96k static OpenImages packing, U-ViT skips, bf16 dmuon.

    Qwen keeps each image's aspect ratio while constraining its pixel area to
    at most 1024x1024. The post-merge token ceiling is 1024 per row; the
    collator fills the budget (static capacity minus one max-size item for the
    isolated padding document) with a variable number of rows.
    """
    static_sequence_length = 3 * _STATIC_SEQUENCE_LENGTH // 2
    token_budget = static_sequence_length - _STATIC_QWEN_MAX_TOKENS_PER_ITEM
    # The Qwen row processor is CPU-heavy; fan it out over spawned worker
    # processes so it does not starve the trainer's main thread.
    num_processor_workers = 4
    merge_size = 2
    encoder = RAEEncoderConfig(
        kind="qwen",
        name="~/models/Qwen3.5-0.8B",
        latent_dim=1024,
        image_size=-1,
        layer_indices=(2, 5, 8, 11),
        merge_size=merge_size,
        # bf16 + flash varlen already gives ~7x over the fp32 sdpa default.
        dtype="bfloat16",
        # Pad the packed encoder input with one isolated document so the
        # native Qwen ViT (encoder.py, bitwise-parity with the HF tower)
        # compiles with fully static shapes. The budget is in post-merge
        # tokens; the encoder consumes pre-merge patches (merge_size**2 x).
        # Without the static shape, torch.compile re-specializes on each
        # batch's packed token count and grid-row count through HF's
        # data-dependent graph breaks, causing a recompile storm across ranks.
        pad_tokens_to=token_budget * merge_size**2,
        # The longest varlen segment (padding doc included) stays below one
        # max-size row: packing slack is always < max_tokens_per_item.
        max_tokens_per_doc=_STATIC_QWEN_MAX_TOKENS_PER_ITEM * merge_size**2,
        compile=True,
    )
    model_spec = model_registry(
        "base",
        attention_backend="varlen",
        latent_dim=encoder.latent_dim,
        decoder_image_size=-1,
        static_sequence_length=static_sequence_length,
        long_skip_connections=((0, 7), (1, 6), (2, 5), (3, 4)),
        use_dmuon=True,
    )
    # Post-merger documents are capped at 1024 tokens by max_pixels, which
    # bounds the attention context for the decoder FLOPs estimate.
    model_spec.model.flops_attention_context = _STATIC_QWEN_MAX_TOKENS_PER_ITEM
    return RAEStage1Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        loss=MSELoss.Config(),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_wandb=True,
            enable_swanlab=True,
            enable_nvml_metrics=True,
        ),
        dataloader=_image_dataloader(
            batch_size=None,
            dataset_path=_OPENIMAGES_TAR_ROOT,
            data_files=_OPENIMAGES_TAR_TRAIN_FILES,
            min_pixels=_STATIC_QWEN_MIN_PIXELS,
            max_pixels=_STATIC_QWEN_MAX_PIXELS,
            token_budget=token_budget,
            max_tokens_per_item=_STATIC_QWEN_MAX_TOKENS_PER_ITEM,
            num_prefetch_batches=4,
            num_processor_workers=num_processor_workers,
            # /mnt/sda1 is a USB-attached NVMe: a single buffered stream tops
            # out far below the device's aggregate bandwidth (the kernel
            # readahead window is latency-bound), so keep the page cache warm
            # with preads.
            readahead_mb=2048,
        ),
        optimizer=_dmuon(2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=625,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=token_budget,
            # token_budget x 8 accumulation microbatches x 4 DP ranks.
            num_tokens_per_train_step=token_budget * 32,
            max_context_length=1,
            max_norm=1.0,
            steps=10000,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            disable_cuda_graphs=False,
        ),
        validator=Validator.Config(
            enable=True,
            # 64 packed validation microbatches per round, every 500 steps:
            # enough images to judge generation quality without stalling
            # training for long.
            steps=64,
            freq=500,
            dataloader=_image_dataloader(
                batch_size=None,
                dataset_path=_OPENIMAGES_VALIDATION_SUBSET_ROOT,
                data_files=_OPENIMAGES_VALIDATION_SUBSET_FILES,
                image_key="image",
                min_pixels=_STATIC_QWEN_MIN_PIXELS,
                max_pixels=_STATIC_QWEN_MAX_PIXELS,
                token_budget=token_budget,
                max_tokens_per_item=_STATIC_QWEN_MAX_TOKENS_PER_ITEM,
                num_prefetch_batches=4,
                num_processor_workers=num_processor_workers,
            ),
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=4,
            data_parallel_shard_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=True, interval=1000),
        # Selective AC measured fastest at the 96k budget (benches 2026-09-09,
        # GAN active from step 0): 55.4 s/step vs 57.4 full; none OOMs at 96k
        # and none at 64k nets lower tokens/s despite fitting.
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["model", "discriminator"]),
        # Four ranks compile the encoder/decoder/discriminator concurrently on
        # the first step, and mid-run CUDA-graph captures pause collectives;
        # loosen the NCCL watchdog bounds accordingly.
        comm=CommConfig(init_timeout_seconds=3600, train_timeout_seconds=600),
        encoder=encoder,
        gan=RAEGANConfig(
            ema_decay=0.9978,
            perceptual_kind="lpips",
            lpips_calibration_checkpoint_path="pretrained_models/lpips/vgg_lpips.pth",
            lpips_vgg_checkpoint_path="pretrained_models/lpips/vgg16-397923af.pth",
            augment=RAEGANAugmentConfig(probability=1.0, cutout=0.0),
            # Half the decoder's dmuon LR: the DINOv3 backbone is frozen, so
            # only the small spectral-norm heads train and 2e-4 over-rotates
            # them.
            discriminator_lr=1e-4,
            discriminator_warmup_steps=625,
            # Earlier than the RAE-paper 0.375/0.5 defaults: at ~407
            # steps/epoch this starts disc training ~2.5 epochs in and the
            # adversarial term ~3.75 epochs in.
            discriminator_update_start_fraction=0.25,
            discriminator_start_fraction=0.375,
            # The discriminator has fully separated real/fake by the time the
            # adversarial term starts, so ramp its weight in instead of taking
            # the full gradient spike on the first GAN step.
            discriminator_weight_ramp_steps=625,
            # Per-patch DINOv3 feature matching joins the adversarial term
            # under the same adaptive weight.
            feature_matching_weight=1.0,
            # L1 cannot see the 16px decoder patch grid (measured ~1.8x
            # ground-truth gradient energy at patch boundaries through step
            # 2000); match spatial gradients so seam crossings answer to the
            # ground truth. 2.0 keeps the term below L1 at convergence
            # (gradient magnitudes are ~10x smaller than pixel errors).
            gradient_loss_weight=2.0,
            # ViTok-v2-style DINOv3 perceptual term (token-wise L2-normalized
            # MSE) once the GAN phase runs. ViTok-v2 uses 500-1000 in its
            # GAN-free, LPIPS-free recipe where this is the only perceptual
            # signal; here it is auxiliary alongside LPIPS + the adversarial
            # term, and the token-normalized MSE is O(1e-4..1e-3), so 100
            # lands the contribution around the pixel loss.
            dinov3_perceptual_weight=100.0,
        ),
        discriminator=RAEFeatureDiscriminator.Config(
            feature_channels=768,
            backbone_kind="hf",
            hf_model_path="~/models/dinov3-vitb16-pretrain-lvd1689m",
            hf_key_depths=(2, 5, 8, 11),
            backbone_batch_size=64,
            # bf16 backbone+heads: disc phase 20.0 -> 12.8 s/step and
            # 84 -> 68 GiB (bench 2026-09-09); feature-space parity vs fp32 is
            # ~1e-2 rel.
            backbone_dtype="bfloat16",
        ),
    )


__all__ = [
    "model_registry",
    "rae_stage1_debug",
    "rae_stage1_openimages_static_96k_uvit",
]
