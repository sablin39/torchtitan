from __future__ import annotations

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
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.protocols.model_spec import ModelSpec
from .data import RAEImageCollator, RAEImageProcessor, RAEQwenCollator, RAEQwenProcessor
from .discriminator import RAEFeatureDiscriminator
from .encoder import RAEEncoderConfig
from .model import RAEDecoder
from .parallelize import parallelize_rae
from .trainer import RAEGANAugmentConfig, RAEGANConfig, RAEStage1Trainer


def model_registry(
    flavor: str = "base", *, attention_backend: str = "sdpa"
) -> ModelSpec:
    if flavor not in {"base", "debug"}:
        raise ValueError(f"Unknown RAE flavor: {flavor}")
    debug = flavor == "debug"
    model = RAEDecoder.Config(
        latent_dim=32 if debug else 768,
        image_size=32 if debug else 256,
        patch_size=8 if debug else 16,
        hidden_size=64 if debug else 512,
        num_layers=2 if debug else 8,
        num_heads=4 if debug else 16,
        intermediate_size=128 if debug else 2048,
        attention_backend=attention_backend,
        spatial_merge_size=2,
        temporal_patch_size=2,
        use_dmuon=debug,
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
    image_size: int | None,
    batch_size: int,
    qwen_model_name: str | None = None,
) -> GrainDataLoader.Config:
    dataset = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="tests/assets/cc12m_test",
            split="train",
            load_dataset_kwargs={"data_files": {"train": "*.tar"}},
        ),
        processor=(
            RAEQwenProcessor.Config(
                model_name=qwen_model_name,
                image_key="jpg",
                image_size=image_size,
            )
            if qwen_model_name is not None
            else RAEImageProcessor.Config(
                image_size=image_size,
                image_key="jpg",
            )
        ),
    )
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=(
            RAEQwenCollator.Config(batch_size=batch_size, media_kind="image")
            if qwen_model_name is not None
            else RAEImageCollator.Config(batch_size=batch_size)
        ),
        repeat=True,
        shuffle=True,
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
    image_size = 256
    batch_size = 2
    config = RAEStage1Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debug", attention_backend="varlen"),
        loss=MSELoss.Config(),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=_image_dataloader(
            image_size=None,
            batch_size=batch_size,
            qwen_model_name="~/models/Qwen3.5-0.8B",
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
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            data_parallel_replicate_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        encoder=RAEEncoderConfig(
            kind="qwen",
            name="~/models/Qwen3.5-0.8B",
            latent_dim=1024,
            image_size=image_size,
            merge_size=2,
        ),
        gan=RAEGANConfig(
            discriminator_start_step=0,
            discriminator_update_start_step=0,
            perceptual_start_step=0,
        ),
    )
    config.model_spec.model.latent_dim = 1024
    config.model_spec.model.image_size = image_size // 2
    return config


def rae_stage1_dmuon() -> RAEStage1Trainer.Config:
    config = rae_stage1_debug()
    config.model_spec = model_registry("base")
    config.model_spec.model.use_dmuon = True
    config.model_spec.model.latent_dim = 1024
    config.model_spec.model.image_size = 128
    config.encoder = RAEEncoderConfig(
        kind="qwen",
        name="~/models/Qwen3.5-0.8B",
        latent_dim=1024,
        image_size=256,
        layer_indices=(),
        merge_size=2,
    )
    config.dataloader = _image_dataloader(
        image_size=None,
        batch_size=1,
        qwen_model_name="~/models/Qwen3.5-0.8B",
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
    config.gan = RAEGANConfig(
        discriminator_start_step=8,
        discriminator_update_start_step=6,
        perceptual_start_step=0,
        ema_decay=0.9978,
        perceptual_kind="lpips",
        lpips_calibration_checkpoint_path="pretrained_models/lpips/vgg_lpips.pth",
        lpips_vgg_checkpoint_path="pretrained_models/lpips/vgg16-397923af.pth",
        augment=RAEGANAugmentConfig(probability=1.0, cutout=0.0),
    )
    config.discriminator = RAEFeatureDiscriminator.Config(
        feature_channels=1024,
        backbone_kind="hf",
        hf_model_path="~/models/dinov3-vitl16-pretrain-lvd1689m",
        hf_key_depths=(5, 11, 17, 23),
    )
    config.parallelism = ParallelismConfig(data_parallel_shard_degree=-1)
    config.checkpoint = CheckpointManager.Config(enable=True, interval=1000)
    return config


__all__ = ["model_registry", "rae_stage1_debug", "rae_stage1_dmuon"]
