# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RAE Stage 1 training loop, augmentation, and metric logging."""

from __future__ import annotations

# Tensor dimensions: B=batch, C=channel, H=height, W=width.

import gc
import math
import os
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.parallel import DistributedDataParallel

from torchtitan.components.checkpointer import LR_SCHEDULER, MODEL, OPTIMIZER
from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.optimizer.dmuon import load_dmuon
from torchtitan.components.validate import BaseValidator, iterate_and_close_dataloader
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .data import RAEQwenCollator
from .decoder import create_rae_static_varlen_metadata
from .discriminator import (
    gan_discriminator_loss,
    gan_generator_loss,
    gan_logits_mean,
    gan_logits_per_image,
    RAEFeatureDiscriminator,
    RAEPerceptualLoss,
)
from .encoder import FrozenRAEEncoder, RAEEncoderConfig


@dataclass
class AugmentationParams:
    """Sampled DiscriminatorAugmentation draws for a set of images.

    ``gates`` holds the three per-call on/off draws (translate, color,
    cutout). ``uniforms`` holds the per-image uniforms from which pixel
    offsets and color factors are derived at apply time, so one params set
    can serve images of different resolutions and can be shared between a
    real image and its paired reconstruction.
    """

    gates: torch.Tensor
    uniforms: torch.Tensor


class DiscriminatorAugmentation:
    """Differentiable translation, color, and cutout augmentation."""

    def __init__(self, probability: float = 1.0, cutout: float = 0.0) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmentation probability must be in [0, 1]")
        if not 0.0 <= cutout <= 1.0:
            raise ValueError("augmentation cutout must be in [0, 1]")
        self.probability = probability
        self.cutout = cutout
        self._grids: dict[
            tuple[int, int, int, torch.device], tuple[torch.Tensor, ...]
        ] = {}

    def _get_grids(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        key = (batch_size, height, width, device)
        if key not in self._grids:
            self._grids[key] = torch.meshgrid(
                torch.arange(batch_size, dtype=torch.long, device=device),
                torch.arange(height, dtype=torch.long, device=device),
                torch.arange(width, dtype=torch.long, device=device),
                indexing="ij",
            )
        return self._grids[key]

    def sample_params(
        self, num_images: int, device: torch.device
    ) -> AugmentationParams:
        """Draw one set of augmentation parameters for ``num_images`` images."""
        gates = torch.rand(3, device=device) <= self.probability
        random_B = torch.rand(7, num_images, 1, 1, device=device)
        return AugmentationParams(gates=gates, uniforms=random_B)

    def apply(
        self, images_BCHW: torch.Tensor, params: AugmentationParams
    ) -> torch.Tensor:
        if images_BCHW.dtype != torch.float32:
            images_BCHW = images_BCHW.float()
        if self.probability < 1e-6:
            return images_BCHW

        apply_translate, apply_color, apply_cutout = params.gates
        batch_size, _, height, width = images_BCHW.shape
        random_B = params.uniforms

        height_delta = round(height * 0.125)
        width_delta = round(width * 0.125)
        height_offset_B11 = (
            random_B[0].mul(2 * height_delta + 1).floor().long() - height_delta
        )
        width_offset_B11 = (
            random_B[1].mul(2 * width_delta + 1).floor().long() - width_delta
        )
        batch_grid_BHW, height_grid_BHW, width_grid_BHW = self._get_grids(
            batch_size, height, width, images_BCHW.device
        )
        height_grid_BHW = (
            (height_grid_BHW + height_offset_B11).add(1).clamp(0, height + 1)
        )
        width_grid_BHW = (width_grid_BHW + width_offset_B11).add(1).clamp(0, width + 1)
        padded_BCHW = F.pad(images_BCHW, (1, 1, 1, 1))
        translated_BCHW = padded_BCHW.permute(0, 2, 3, 1)[
            batch_grid_BHW,
            height_grid_BHW,
            width_grid_BHW,
        ].permute(0, 3, 1, 2)
        images_BCHW = torch.where(apply_translate, translated_BCHW, images_BCHW)

        colored_BCHW = images_BCHW + random_B[2].unsqueeze(-1) - 0.5
        channel_mean_B1HW = colored_BCHW.mean(dim=1, keepdim=True)
        colored_BCHW = (colored_BCHW - channel_mean_B1HW) * random_B[3].unsqueeze(
            -1
        ).mul(2) + channel_mean_B1HW
        image_mean_B111 = colored_BCHW.mean((1, 2, 3), keepdim=True)
        colored_BCHW = (colored_BCHW - image_mean_B111) * random_B[4].unsqueeze(-1).add(
            0.5
        ) + image_mean_B111
        images_BCHW = torch.where(apply_color, colored_BCHW, images_BCHW)

        if self.cutout > 0:
            cutout_height = round(height * self.cutout)
            cutout_width = round(width * self.cutout)
            height_offset_B11 = (
                random_B[5].mul(height + (1 - cutout_height % 2)).floor().long()
            )
            width_offset_B11 = (
                random_B[6].mul(width + (1 - cutout_width % 2)).floor().long()
            )
            batch_grid_BHW, height_grid_BHW, width_grid_BHW = self._get_grids(
                batch_size,
                cutout_height,
                cutout_width,
                images_BCHW.device,
            )
            height_grid_BHW = (
                (height_grid_BHW + height_offset_B11)
                .sub(cutout_height // 2)
                .clamp(0, height - 1)
            )
            width_grid_BHW = (
                (width_grid_BHW + width_offset_B11)
                .sub(cutout_width // 2)
                .clamp(0, width - 1)
            )
            mask_BHW = torch.ones(
                batch_size,
                height,
                width,
                dtype=images_BCHW.dtype,
                device=images_BCHW.device,
            )
            mask_BHW[
                batch_grid_BHW, height_grid_BHW, width_grid_BHW
            ] = images_BCHW.new_zeros(())
            cutout_BCHW = images_BCHW * mask_BHW.unsqueeze(1)
            images_BCHW = torch.where(apply_cutout, cutout_BCHW, images_BCHW)

        return images_BCHW.contiguous()

    def __call__(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        if images_BCHW.dtype != torch.float32:
            images_BCHW = images_BCHW.float()
        if self.probability < 1e-6:
            return images_BCHW
        return self.apply(
            images_BCHW,
            self.sample_params(images_BCHW.shape[0], images_BCHW.device),
        )


def log_stage1_metrics(
    step: int,
    losses: Sequence[torch.Tensor],
    *,
    metrics_processor: Any | None = None,
    non_padding_ratio: float = 1.0,
    num_images_per_step: float = 0.0,
    epoch: float | None = None,
    tokens_last_epoch: int | None = None,
) -> None:
    """Log the scalar metrics emitted by one RAE Stage 1 update."""
    values = [float(loss.detach().item()) for loss in losses]
    if len(values) != 13:
        raise ValueError(f"RAE Stage 1 metrics require 13 values, got {len(values)}")
    if metrics_processor is not None:
        extra_metrics: dict[str, float] = {
            "rae/reconstruction_loss": values[0],
            "rae/perceptual_loss": values[1],
            "rae/adversarial_loss": values[2],
            "rae/discriminator_loss": values[3],
            "rae/adaptive_weight": values[4],
            "rae/decoder_grad_norm": values[5],
            "rae/discriminator_grad_norm": values[6],
            "rae/generator_logit": values[7],
            "rae/discriminator_real_logit": values[8],
            "rae/discriminator_fake_logit": values[9],
            "rae/discriminator_accuracy": values[10],
            "rae/dino_feature_distance": values[11],
            "rae/feature_matching_loss": values[12],
            "rae/non_padding_ratio": non_padding_ratio,
            "rae/num_images_per_step": num_images_per_step,
        }
        if epoch is not None:
            extra_metrics["rae/epoch"] = epoch
        if tokens_last_epoch is not None:
            extra_metrics["rae/tokens_last_epoch"] = float(tokens_last_epoch)
        metrics_processor.log(
            step,
            global_avg_loss=values[0],
            global_max_loss=values[0],
            grad_norm=values[5],
            extra_metrics=extra_metrics,
        )
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        epoch_suffix = f" epoch={int(epoch)}" if epoch is not None else ""
        logger.info(
            "[RAE Stage 1 | step %d] recon=%.5f perceptual=%.5f "
            "gan=%.5f disc=%.5f adaptive=%.5f decoder_grad=%.5f "
            "disc_grad=%.5f gen_logit=%.5f real_logit=%.5f fake_logit=%.5f "
            "disc_acc=%.5f dino_dist=%.5f fm=%.5f non_padding=%.5f "
            "images_per_step=%.2f%s",
            step,
            *values,
            non_padding_ratio,
            num_images_per_step,
            epoch_suffix,
        )


ImageBatch = torch.Tensor | list[torch.Tensor]


class _PhaseProfiler:
    """Per-step phase timing: CPU wall time plus CUDA-event spans.

    Enabled by the RAE_PROFILE_PHASES=1 environment variable. CUDA events are
    recorded without synchronizing and read back once per step, so profiling
    overhead is negligible. The GPU span of a phase includes stream idle time
    spent waiting on the CPU, which is what exposes launch-bound phases.
    """

    def __init__(self, device: torch.device, *, enabled: bool) -> None:
        self.enabled = enabled and device.type == "cuda"
        self._device = device
        self._cpu_totals: dict[str, float] = defaultdict(float)
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._open: dict[str, Any] = {}

    @contextmanager
    def phase(self, name: str) -> Generator[None]:
        if not self.enabled:
            yield
            return
        start_cpu = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))
            self._cpu_totals[name] += time.perf_counter() - start_cpu

    def record(self, name: str, cpu_seconds: float) -> None:
        """Add externally measured CPU seconds to a phase's step total."""
        if self.enabled:
            self._cpu_totals[name] += cpu_seconds

    def collect(self) -> dict[str, tuple[float, float]]:
        """Return {phase: (cpu_seconds, gpu_seconds)} and reset for the step."""
        if not self.enabled:
            return {}
        if self._pending:
            torch.cuda.synchronize(self._device)
        gpu_totals: dict[str, float] = defaultdict(float)
        for name, start, end in self._pending:
            gpu_totals[name] += start.elapsed_time(end) / 1e3
        self._pending.clear()
        report = {
            name: (cpu, gpu_totals.get(name, 0.0))
            for name, cpu in self._cpu_totals.items()
        }
        self._cpu_totals.clear()
        return report

    def start(self, name: str) -> None:
        """Open a phase without a with-block; pair with ``stop``."""
        manager = self.phase(name)
        manager.__enter__()
        self._open[name] = manager

    def stop(self, name: str) -> None:
        self._open.pop(name).__exit__(None, None, None)


@dataclass(frozen=True, slots=True)
class RAEGANAugmentConfig:
    """RAEv2 DiffAug settings for discriminator inputs."""

    probability: float = 1.0
    cutout: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("gan.augment.probability must be in [0, 1]")
        if not 0.0 <= self.cutout <= 1.0:
            raise ValueError("gan.augment.cutout must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RAEGANConfig:
    """GAN schedule and loss settings.

    RAEv2 expresses the GAN phase boundaries in epochs: discriminator updates
    begin at 0.375 and the adversarial generator loss at 0.5 of training, with
    L1+LPIPS active from the start. The ``*_fraction`` fields reproduce that
    schedule for any ``training.steps``; the ``*_step`` fields pin absolute
    step boundaries instead when set.
    """

    discriminator_start_step: int | None = None
    discriminator_update_start_step: int | None = None
    discriminator_start_fraction: float = 0.5
    discriminator_update_start_fraction: float = 0.375
    perceptual_start_step: int = 0
    discriminator_weight: float = 0.75
    discriminator_weight_ramp_steps: int = 0
    """Steps over which the generator-side GAN weight ramps linearly from 0
    to ``discriminator_weight`` after the adversarial phase starts. The
    discriminator has already converged by then, so a full-weight first step
    spikes the generator gradient; 0 keeps RAEv2's step-function onset."""
    feature_matching_weight: float = 1.0
    """Weight of the per-patch DINOv3 feature-matching term folded into the
    adaptive-weighted adversarial loss; 0 disables it."""
    perceptual_weight: float = 1.0
    discriminator_updates: int = 1
    discriminator_update_batch_size: int = 256
    """Images per backward in one discriminator update. A packed step holds
    ~1000 native-resolution images; backward over all of them at once retains
    the whole backbone/head activation graph and OOMs, so the update chunks
    the image list and weights each chunk's mean loss by its image count."""
    generator_loss: str = "vanilla"
    discriminator_loss: str = "hinge"
    max_adaptive_weight: float = 10000.0
    ema_decay: float = 0.9995
    discriminator_lr: float = 2e-4
    discriminator_betas: tuple[float, float] = (0.9, 0.95)
    discriminator_weight_decay: float = 0.0
    discriminator_warmup_steps: int = 0
    discriminator_final_lr_ratio: float = 0.1
    perceptual_kind: str = "fixed"
    perceptual_resize_long_side: int = 256
    """Long-side bound for perceptual-loss inputs; 0 keeps native resolution.
    VGG/LPIPS is calibrated around 224-256px, so larger reconstructions are
    downscaled (aspect preserved) before feature extraction. Native-resolution
    LPIPS was measured at ~16-23 s/step of the ~50 s stage-1 step."""
    lpips_calibration_checkpoint_path: str = ""
    lpips_vgg_checkpoint_path: str | None = None
    augment: RAEGANAugmentConfig = field(default_factory=RAEGANAugmentConfig)

    def __post_init__(self) -> None:
        if self.discriminator_updates <= 0:
            raise ValueError("gan.discriminator_updates must be positive")
        if self.discriminator_weight_ramp_steps < 0:
            raise ValueError("gan.discriminator_weight_ramp_steps must be non-negative")
        if self.discriminator_update_batch_size <= 0:
            raise ValueError("gan.discriminator_update_batch_size must be positive")
        if self.discriminator_weight < 0 or self.perceptual_weight < 0:
            raise ValueError("GAN and perceptual weights must be non-negative")
        if self.feature_matching_weight < 0:
            raise ValueError("gan.feature_matching_weight must be non-negative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("gan.ema_decay must be in [0, 1)")
        for name in (
            "discriminator_start_step",
            "discriminator_update_start_step",
            "perceptual_start_step",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"gan.{name} must be non-negative or None")
        if not 0.0 <= self.discriminator_start_fraction <= 1.0:
            raise ValueError("gan.discriminator_start_fraction must be in [0, 1]")
        if not 0.0 <= self.discriminator_update_start_fraction <= 1.0:
            raise ValueError(
                "gan.discriminator_update_start_fraction must be in [0, 1]"
            )
        if not 0.0 < self.discriminator_final_lr_ratio <= 1.0:
            raise ValueError("gan.discriminator_final_lr_ratio must be in (0, 1]")
        if self.generator_loss not in {"hinge", "vanilla"}:
            raise ValueError(f"Unsupported generator GAN loss: {self.generator_loss}")
        if self.discriminator_loss not in {"hinge", "vanilla"}:
            raise ValueError(
                f"Unsupported discriminator GAN loss: {self.discriminator_loss}"
            )
        if self.perceptual_kind not in {"fixed", "lpips", "dinov3"}:
            raise ValueError(
                f"Unsupported perceptual loss kind: {self.perceptual_kind}"
            )
        if self.perceptual_resize_long_side < 0:
            raise ValueError("gan.perceptual_resize_long_side must be non-negative")
        if (
            self.perceptual_kind == "lpips"
            and not self.lpips_calibration_checkpoint_path
        ):
            raise ValueError(
                "gan.lpips_calibration_checkpoint_path is required for LPIPS"
            )

    def discriminator_update_start(self, total_steps: int) -> int:
        if self.discriminator_update_start_step is not None:
            return self.discriminator_update_start_step
        return round(self.discriminator_update_start_fraction * max(total_steps, 1))

    def discriminator_start(self, total_steps: int) -> int:
        if self.discriminator_start_step is not None:
            return self.discriminator_start_step
        return round(self.discriminator_start_fraction * max(total_steps, 1))


class _RAEModelState(Stateful):
    def __init__(
        self,
        decoder: nn.Module,
        discriminator: nn.Module,
        ema: nn.Module,
    ):
        self.decoder = decoder
        self.discriminator = discriminator
        self.ema = ema

    def state_dict(self) -> dict[str, Any]:
        decoder_state = self.decoder.state_dict()
        if getattr(self.decoder, "_dmuon_enabled", False) and hasattr(
            self.decoder, "_dedicated_comm_ctx"
        ):
            dmuon = load_dmuon()

            decoder_state = dmuon.get_model_state_dict(
                self.decoder, cpu_offload=False, rank0_only=False
            )
            decoder_state.update(
                {
                    name: buffer.detach().clone()
                    for name, buffer in self.decoder.named_buffers()
                }
            )
        ema_state = self.ema.state_dict()
        return {
            "decoder": decoder_state,
            "discriminator": self.discriminator.state_dict(),
            "ema": ema_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Strict everywhere: a silently skipped key leaves randomly
        # initialized weights behind (a past discriminator key rename would
        # have done exactly that under strict=False).
        decoder_state = state_dict.get("decoder", state_dict)
        if getattr(self.decoder, "_dmuon_enabled", False) and hasattr(
            self.decoder, "_dedicated_comm_ctx"
        ):
            dmuon = load_dmuon()

            dmuon.set_model_state_dict(self.decoder, decoder_state)
            for name, buffer in self.decoder.named_buffers():
                if name not in decoder_state:
                    raise RuntimeError(
                        f"RAE checkpoint is missing decoder buffer: {name}"
                    )
                buffer.copy_(
                    decoder_state[name].to(device=buffer.device, dtype=buffer.dtype)
                )
        else:
            self.decoder.load_state_dict(decoder_state, strict=True)
        self.discriminator.load_state_dict(state_dict["discriminator"], strict=True)
        self.ema.load_state_dict(state_dict["ema"], strict=True)


class _RAEOptimizerState(Stateful):
    def __init__(self, generator, discriminator: torch.optim.Optimizer) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "generator" in state_dict:
            self.generator.load_state_dict(state_dict["generator"])
        if "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"])


class _RAESchedulerState(Stateful):
    def __init__(self, generator: LRSchedulersContainer, discriminator) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "generator" in state_dict:
            self.generator.load_state_dict(state_dict["generator"])
        if "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"])


class RAEStage1Trainer(Trainer):
    """TorchTitan trainer implementing the alternating RAEv2 GAN phases."""

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        encoder: RAEEncoderConfig = field(default_factory=RAEEncoderConfig)
        gan: RAEGANConfig = field(default_factory=RAEGANConfig)
        discriminator: RAEFeatureDiscriminator.Config = field(
            default_factory=RAEFeatureDiscriminator.Config
        )
        epochs: int | None = None
        """Stop once every rank's data stream has completed this many epochs.

        None keeps step-only training. The LR/GAN schedule horizon stays
        ``training.steps`` regardless; epochs only terminates the run.
        """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.validator = RAEValidator(config.validator, self)
        decoder = self.model_parts[0]
        static_sequence_length = int(getattr(decoder, "static_sequence_length", 0))
        if not config.training.disable_cuda_graphs and (
            self.device.type != "cuda" or static_sequence_length <= 0
        ):
            raise ValueError(
                "RAE CUDA graphs require a CUDA device and a positive "
                "decoder static_sequence_length"
            )
        self._static_sequence_length = static_sequence_length
        self.encoder = FrozenRAEEncoder(config.encoder, self.device)
        if (
            decoder.image_size != -1
            and self.encoder.supervision_image_size is not None
            and decoder.image_size != self.encoder.supervision_image_size
        ):
            raise ValueError(
                "RAE decoder image_size must equal encoder image_size / merge_size: "
                f"{decoder.image_size} != {self.encoder.supervision_image_size}"
            )
        self.discriminator = RAEFeatureDiscriminator(
            config.discriminator,
            device=self.device,
        ).to(self.device)
        if config.compile.enable and "discriminator" in config.compile.components:
            self.discriminator.compile_forward(backend=config.compile.backend)
        if (
            dist.is_available()
            and dist.is_initialized()
            and self.parallel_dims.dp_enabled
        ):
            ddp_kwargs = {}
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.device.index]
            self.discriminator_train = DistributedDataParallel(
                self.discriminator,
                broadcast_buffers=False,
                find_unused_parameters=True,
                **ddp_kwargs,
            )
        else:
            self.discriminator_train = self.discriminator

        if config.gan.perceptual_kind == "dinov3":
            if config.discriminator.backbone_kind != "hf":
                raise ValueError(
                    "gan.perceptual_kind='dinov3' requires "
                    "discriminator.backbone_kind='hf'"
                )
            self.perceptual_loss = None
        else:
            self.perceptual_loss = RAEPerceptualLoss(
                kind=config.gan.perceptual_kind,
                channels=config.discriminator.feature_channels,
                calibration_checkpoint_path=config.gan.lpips_calibration_checkpoint_path,
                vgg_checkpoint_path=config.gan.lpips_vgg_checkpoint_path,
                resize_long_side=config.gan.perceptual_resize_long_side,
            ).to(self.device)
        self.discriminator_augmentation = DiscriminatorAugmentation(
            probability=config.gan.augment.probability,
            cutout=config.gan.augment.cutout,
        )
        warmup_discriminator = (
            config.compile.enable
            and "discriminator" in config.compile.components
            and config.discriminator.backbone_kind == "hf"
        )
        if warmup_discriminator:
            autocast_bf16 = (
                self.device.type == "cuda" and config.training.dtype == "bfloat16"
            )
            dummy_items = [
                torch.zeros(
                    3,
                    256,
                    256,
                    device=self.device,
                    dtype=torch.bfloat16 if autocast_bf16 else torch.float32,
                )
                for _ in range(config.discriminator.backbone_batch_size)
            ]
        if warmup_discriminator and config.gan.discriminator_weight > 0:
            # Warm both autograd variants of the compiled discriminator during
            # initialization (covered by the init timeout): the first GAN step
            # would otherwise compile mid-run, and slow per-rank compilation
            # can stall the DP collectives. The warmups must replicate the
            # runtime guards (frozen vs trainable heads, bf16 autocast,
            # grad-enabled generator-side inputs, CHW list input) or dynamo
            # compiles separate variants and the warmup is wasted. The forward
            # is compiled with dynamic shapes, so one warmup per variant
            # covers the native resolutions seen at runtime.
            # Generator-side variant: frozen heads, grad-enabled inputs,
            # forward+backward (the backward compiles the backbone's gradient
            # graph, which the first GAN step would otherwise build mid-run).
            self.discriminator.eval()
            self.discriminator.set_head_requires_grad(False)
            generator_dummies = [item.requires_grad_(True) for item in dummy_items]
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_bf16,
            ):
                generator_logits = self.discriminator(
                    self._augment_images(generator_dummies)
                )
            torch.stack([logits.sum() for logits in generator_logits]).sum().backward()
            # Discriminator-side variant: trainable heads, forward+backward.
            self.discriminator.set_head_requires_grad(True)
            self.discriminator.train()
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_bf16,
            ):
                warmup_logits = self.discriminator(self._augment_images(dummy_items))
            torch.stack([logits.sum() for logits in warmup_logits]).sum().backward()
            self.discriminator.zero_grad(set_to_none=True)
            if config.gan.feature_matching_weight > 0:
                # The step-start real-feature cache calls the compiled backbone
                # in a no-grad variant; warm it here so the first
                # feature-matching step does not compile mid-run.
                with torch.no_grad():
                    self.discriminator.cache_real_features(dummy_items)
            self.discriminator.eval()
            self.discriminator.set_head_requires_grad(False)
        if warmup_discriminator and config.gan.perceptual_kind == "dinov3":
            # The DINOv3 perceptual term runs from step 0 through its own
            # features-only autograd variants (no-grad reals, grad-carrying
            # fakes, backward through the feature-matching reduction); warm
            # them here or the first perceptual step compiles mid-run.
            self.discriminator.eval()
            self.discriminator.set_head_requires_grad(False)
            perceptual_dummies = [
                item.detach().clone().requires_grad_(True) for item in dummy_items
            ]
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_bf16,
            ):
                warmup_perceptual = self._perceptual_loss(
                    dummy_items, perceptual_dummies
                )
            warmup_perceptual.backward()
            self.discriminator.zero_grad(set_to_none=True)
            self.discriminator.eval()
            self.discriminator.set_head_requires_grad(False)

        if config.epochs is not None and config.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {config.epochs}")
        self._phase_profiler = _PhaseProfiler(
            self.device, enabled=os.environ.get("RAE_PROFILE_PHASES") == "1"
        )
        self._step_end_time: float | None = None
        if self._phase_profiler.enabled:
            # Surface full-GC pauses in the per-step report: a multi-second
            # gen2 collection stalls the whole rank and (via DP collectives)
            # paces every other rank.
            gc_start: dict[str, float] = {}

            def _gc_callback(phase: str, info: dict[str, int]) -> None:
                if info["generation"] != 2:
                    return
                if phase == "start":
                    gc_start["t"] = time.perf_counter()
                elif phase == "stop" and "t" in gc_start:
                    self._phase_profiler.record(
                        "gc", time.perf_counter() - gc_start.pop("t")
                    )

            gc.callbacks.append(_gc_callback)
        self._tokens_current_epoch = 0
        self._last_seen_epoch = 0
        self._tokens_last_epoch: int | None = None
        self._warned_epoch_tracking_missing = False
        self._warned_adaptive_weight_unavailable = False
        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.gan.discriminator_lr,
            betas=config.gan.discriminator_betas,
            weight_decay=config.gan.discriminator_weight_decay,
        )
        self.disc_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.disc_optimizer,
            self._make_discriminator_schedule(config),
        )
        self.ema_model = self._build_ema(decoder)

        if getattr(self.checkpointer, "enable", False):
            self.checkpointer.states[MODEL] = _RAEModelState(
                decoder,
                self.discriminator,
                self.ema_model,
            )
            self.checkpointer.states[OPTIMIZER] = _RAEOptimizerState(
                self.optimizers, self.disc_optimizer
            )
            self.checkpointer.states[LR_SCHEDULER] = _RAESchedulerState(
                self.lr_schedulers, self.disc_scheduler
            )

    @staticmethod
    def _make_discriminator_schedule(config: Config):
        """RAEv2-style linear warmup followed by a cosine decay to the final ratio."""
        warmup_steps = max(0, config.gan.discriminator_warmup_steps)
        total_steps = max(
            1, config.training.steps * max(1, config.gan.discriminator_updates)
        )
        final_ratio = config.gan.discriminator_final_lr_ratio

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(
                max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0), 1
            )
            return final_ratio + (1.0 - final_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return schedule

    def _build_ema(self, decoder: nn.Module) -> nn.Module:
        config = getattr(decoder, "config", None)
        if config is None:
            raise RuntimeError(
                "RAE EMA requires the decoder to expose its config for an "
                "unsharded EMA copy."
            )

        ema_model = config.build().to(self.device)
        ema_model.init_states()
        if getattr(decoder, "_dmuon_enabled", False) and hasattr(
            decoder, "_dedicated_comm_ctx"
        ):
            dmuon = load_dmuon()
            state = dmuon.get_model_state_dict(
                decoder, cpu_offload=False, rank0_only=False
            )
        else:
            state = decoder.state_dict()
        normalized_state = {
            name.replace("._checkpoint_wrapped_module.", "."): value
            for name, value in state.items()
        }
        ema_model.load_state_dict(normalized_state, strict=True)
        ema_model.eval()
        ema_model.requires_grad_(False)
        return ema_model

    @torch.no_grad()
    def _update_ema(self) -> None:
        decoder = self.model_parts[0]
        decay = self.config.gan.ema_decay
        if getattr(decoder, "_dmuon_enabled", False) and hasattr(
            decoder, "_dedicated_comm_ctx"
        ):
            dmuon = load_dmuon()
            state = dmuon.get_model_state_dict(
                decoder, cpu_offload=False, rank0_only=False
            )
            for name, ema_parameter in self.ema_model.named_parameters():
                parameter = state.get(name)
                if parameter is None and name.startswith("layers."):
                    layer_prefix, layer_id, remainder = name.split(".", 2)
                    wrapped_name = (
                        f"{layer_prefix}.{layer_id}._checkpoint_wrapped_module."
                        f"{remainder}"
                    )
                    parameter = state.get(wrapped_name)
                if parameter is None:
                    raise RuntimeError(
                        f"DMuon model state is missing EMA parameter {name!r}"
                    )
                ema_parameter.mul_(decay).add_(
                    parameter.to(
                        device=ema_parameter.device,
                        dtype=ema_parameter.dtype,
                    ).detach(),
                    alpha=1.0 - decay,
                )
            return
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), decoder.parameters(), strict=True
        ):
            ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    def _decoder_grad_norm(self) -> torch.Tensor:
        parameters = [p for p in self.model_parts[0].parameters() if p.requires_grad]
        grad_norm = dist_utils.clip_grad_norm_(
            parameters,
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=None,
            ep_enabled=False,
        )
        squared_norm = grad_norm.float().square()
        for optimizer in self.optimizers:
            if not hasattr(optimizer, "_dedicated_params"):
                continue
            dmuon = load_dmuon()
            stats = dmuon.clip_grad_norm_(
                optimizer,
                self.config.training.max_norm,
                foreach=True,
            )
            squared_norm = squared_norm + stats.total_norm.float().square()
        return squared_norm.sqrt()

    def _next_images(
        self, data_iterator: Iterator, *, count_training_stats: bool = True
    ) -> tuple[ImageBatch, Mapping[str, Any] | None]:
        try:
            input_dict, labels = next(data_iterator)
        except DataloaderExhaustedError:
            raise
        if "input" not in input_dict:
            raise KeyError("RAE Stage 1 batches must contain an 'input' image tensor")
        # The base batch generator counts one dummy label per sample. RAE MFU is
        # defined per post-merger latent token, so replace that sample count after
        # the encoder exposes the actual runtime grid below.
        self.metrics_processor.ntokens_since_last_log -= labels.numel()
        if count_training_stats:
            self.ntokens_seen += labels.numel()
            self.n_valid_tokens_seen += labels.numel()
            self.n_nonpad_tokens_seen += labels.numel()
        images = input_dict["input"]
        encoder_input: Mapping[str, Any] | None = None
        if "media" in input_dict:
            media = input_dict["media"]
            if not isinstance(media, (list, tuple)):
                raise ValueError("RAE Qwen media must be a list of BTCHW tensors")
            image_items = []
            for media_item in media:
                if media_item.ndim != 5 or media_item.shape[0] != 1:
                    raise ValueError("RAE Qwen media must contain one BTCHW item")
                if media_item.shape[1] != 1:
                    raise ValueError(
                        "RAE Stage 1 image training accepts only one-frame media"
                    )
                image_items.append(media_item[0, 0])
            images = image_items
            encoder_input = {
                name: value
                for name, value in input_dict.items()
                if torch.is_tensor(value)
            }
        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError("RAE Stage 1 input must have BCHW shape")
            images = images.to(self.device, non_blocking=True)
        elif isinstance(images, (list, tuple)):
            images = [image.to(self.device, non_blocking=True) for image in images]
            if not images or any(image.ndim != 3 for image in images):
                raise ValueError("RAE Stage 1 image lists must contain CHW tensors")
        else:
            raise ValueError("RAE Stage 1 input must be a BCHW tensor or CHW list")
        if encoder_input is not None:
            encoder_input = {
                name: value.to(self.device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for name, value in encoder_input.items()
            }
        return images, encoder_input

    def _supervision_images(
        self,
        images_BCHW: ImageBatch,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> ImageBatch:
        if isinstance(images_BCHW, (list, tuple)):
            image_items = list(images_BCHW)
            if target_sizes is None:
                target_size = self.encoder.supervision_image_size
                target_sizes = (
                    [(target_size, target_size) for _ in image_items]
                    if target_size is not None
                    else [(image.shape[-2], image.shape[-1]) for image in image_items]
                )
            if len(target_sizes) != len(image_items):
                raise ValueError("Supervision target sizes must match image count")
            return [
                self._resize_supervision_image(image, size)
                for image, size in zip(image_items, target_sizes, strict=True)
            ]
        if images_BCHW.ndim != 4:
            raise ValueError("RAE supervision expects BCHW images")
        target_size = self.encoder.supervision_image_size
        if target_size is None:
            return images_BCHW.clamp(0, 1)
        if images_BCHW.shape[-2:] == (target_size, target_size):
            return images_BCHW
        return F.interpolate(
            images_BCHW,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)

    @staticmethod
    def _resize_supervision_image(
        image_CHW: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        if image_CHW.ndim != 3:
            raise ValueError("RAE supervision image lists must contain CHW tensors")
        if image_CHW.shape[-2:] == target_size:
            return image_CHW
        return (
            F.interpolate(
                image_CHW.unsqueeze(0),
                size=target_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .squeeze(0)
            .clamp(0, 1)
        )

    @staticmethod
    def _image_items(images: ImageBatch) -> list[torch.Tensor]:
        if isinstance(images, torch.Tensor):
            if images.ndim != 4:
                raise ValueError("RAE image batches must have BCHW shape")
            return list(images.unbind(0))
        return list(images)

    def _static_decode(
        self,
        decoder: nn.Module,
        latents: torch.Tensor,
        grid_thw: torch.Tensor,
        fps: torch.Tensor | float | None,
        temporal_start: torch.Tensor | float,
    ) -> torch.Tensor:
        if self._static_sequence_length <= 0:
            raise ValueError("RAE static decode requires a positive token budget")
        if latents.ndim == 3:
            latents_TD = latents.reshape(-1, latents.shape[-1])
        elif latents.ndim == 2:
            latents_TD = latents
        else:
            raise ValueError(
                "RAE static decode expects packed or batched token latents"
            )
        grid = grid_thw.reshape(-1, 3).to(device=latents.device, dtype=torch.long)
        sequence_lengths = grid.prod(dim=-1)
        valid_length = int(sequence_lengths.sum().item())
        static_length = self._static_sequence_length
        if static_length < valid_length:
            raise ValueError(
                "RAE decoder static_sequence_length is smaller than this batch; "
                f"need at least {valid_length} token slots, got {static_length}"
            )
        latents_static = F.pad(latents_TD, (0, 0, 0, static_length - valid_length))
        rope = decoder.layers[0].attention.rope
        positions = rope.build_packed_positions(
            grid,
            fps=fps,
            temporal_start=temporal_start,
        )
        positions = F.pad(positions, (0, 0, 0, static_length - valid_length))
        metadata = create_rae_static_varlen_metadata(
            sequence_lengths,
            static_length,
            device=latents.device,
        )
        patch_logits = decoder(
            latents_static,
            padded_positions_T3=positions,
            attention_masks=metadata,
            return_padded=True,
        )
        return patch_logits[:valid_length]

    def _encode(
        self,
        images: ImageBatch,
        encoder_input: Mapping[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | float]:
        """Run the frozen encoder once and return clean latents with metadata.

        ``grid_thw`` is the encoder's CPU-side copy: token counts and
        supervision sizes derive from it without synchronizing the GPU, and
        the decoder moves it on-device itself where needed.
        """
        encoder_source = encoder_input if encoder_input is not None else images
        encoded = self.encoder(
            encoder_source,
            return_grid_thw=True,
        )
        if not isinstance(encoded, tuple):
            raise RuntimeError("RAE encoder must return grid metadata for Stage 1")
        latents, _ = encoded
        grid_thw = self.encoder.last_grid_thw
        if grid_thw is None:
            raise RuntimeError("RAE encoder did not expose grid metadata")
        temporal_start = (
            self.encoder.last_temporal_start
            if self.encoder.last_temporal_start is not None
            else 0.0
        )
        return latents.detach(), grid_thw, self.encoder.last_fps, temporal_start

    @staticmethod
    def _add_latent_noise(
        latents: torch.Tensor,
        grid_thw: torch.Tensor,
        noise_tau: float,
    ) -> torch.Tensor:
        """Apply RAE latent noise with one scale per packed media item."""
        if noise_tau <= 0:
            return latents
        if latents.ndim == 2:
            tokens_per_item = (
                grid_thw.reshape(-1, 3).prod(dim=-1).to(device=latents.device)
            )
            if int(tokens_per_item.sum().item()) != latents.shape[0]:
                raise ValueError(
                    "RAE grid metadata does not match the packed latent count"
                )
            noise_scale = torch.repeat_interleave(
                noise_tau
                * torch.rand(
                    tokens_per_item.numel(),
                    device=latents.device,
                    dtype=latents.dtype,
                ),
                tokens_per_item,
            ).unsqueeze(-1)
        elif latents.ndim == 3:
            noise_scale = noise_tau * torch.rand(
                (latents.shape[0], 1, 1), device=latents.device, dtype=latents.dtype
            )
        else:
            noise_scale = noise_tau * torch.rand(
                (latents.shape[0], 1, 1, 1),
                device=latents.device,
                dtype=latents.dtype,
            )
        return latents + noise_scale * torch.randn_like(latents)

    def _decode(
        self,
        decoder: nn.Module,
        latents: torch.Tensor,
        grid_thw: torch.Tensor,
        fps: torch.Tensor | float | None,
        temporal_start: torch.Tensor | float,
    ) -> list[torch.Tensor]:
        if self._static_sequence_length > 0:
            decoded = self._static_decode(
                decoder,
                latents,
                grid_thw,
                fps,
                temporal_start,
            )
        else:
            decoded = decoder(
                latents,
                grid_thw=grid_thw,
                fps=fps,
                temporal_start=temporal_start,
            )
        if latents.ndim == 2 or decoded.ndim == 2:
            return decoder.unpatchify_packed(
                decoded,
                grid_thw,
                patch_size=decoder.patch_size,
            )
        return self._image_items(decoded)

    def _encode_decode(
        self,
        decoder: nn.Module,
        images: ImageBatch,
        encoder_input: Mapping[str, Any] | None,
        *,
        add_noise: bool,
    ) -> list[torch.Tensor]:
        latents, grid_thw, fps, temporal_start = self._encode(images, encoder_input)
        if add_noise:
            latents = self._add_latent_noise(latents, grid_thw, self.encoder.noise_tau)
        return self._decode(decoder, latents, grid_thw, fps, temporal_start)

    def _perceptual_loss(
        self,
        real_items: list[torch.Tensor],
        fake_items: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.config.gan.perceptual_kind == "dinov3":
            # DINOv3 feature-space perceptual term on clean (un-augmented)
            # pairs at native resolution: the real branch runs no-grad, the
            # fake branch carries gradients to the decoder through the frozen
            # backbone. Same per-patch reduction as feature matching, but it
            # enters from step 0 under the fixed perceptual weight.
            with torch.no_grad():
                real_features = self.discriminator.features(
                    [(image + 1.0) * 0.5 for image in real_items],
                    use_compiled=True,
                )
            fake_features = self.discriminator.features(
                [(image + 1.0) * 0.5 for image in fake_items],
                use_compiled=True,
            )
            return self.discriminator.feature_matching(real_features, fake_features)
        # One grouped call per microbatch: equal-shape images share backbone
        # forwards, and the real branch runs under no_grad.
        return self.perceptual_loss.forward_per_sample_list(
            real_items, fake_items
        ).mean()

    def _augment_images(
        self,
        images: list[torch.Tensor],
        *,
        params: AugmentationParams | None = None,
    ) -> list[torch.Tensor]:
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, image in enumerate(images):
            grouped.setdefault((image.shape[-2], image.shape[-1]), []).append(index)
        augmented = list(images)
        for indices in grouped.values():
            group = torch.stack([images[index] for index in indices])
            if params is None:
                group = self.discriminator_augmentation(group)
            else:
                group_params = AugmentationParams(
                    gates=params.gates, uniforms=params.uniforms[:, indices]
                )
                group = self.discriminator_augmentation.apply(group, group_params)
            for group_index, image_index in enumerate(indices):
                augmented[image_index] = group[group_index]
        return augmented

    def _update_discriminator(
        self,
        fake_normed_items: list[torch.Tensor],
        real_normed_items: list[torch.Tensor],
        *,
        real_features: list[list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Chunked discriminator update over native-resolution image lists.

        Each chunk's mean loss is weighted by its image count, so accumulated
        gradients match a single full-batch mean while peak activation memory
        stays bounded by gan.discriminator_update_batch_size. Returns
        (loss, real_logits, fake_logits, accuracy) batch means.

        When ``real_features`` carries the step-start cached backbone
        activations of the same reals (augmented with the step's shared
        augmentation draws, so the real pass keeps its usual augmented
        distribution), the eager heads run on the cache and the real backbone
        forward is not repeated.
        """
        num_images = len(fake_normed_items)
        batch_size = self.config.gan.discriminator_update_batch_size
        sums = [
            torch.zeros((), device=self.device, dtype=torch.float32) for _ in range(4)
        ]
        for start in range(0, num_images, batch_size):
            fake_chunk = fake_normed_items[start : start + batch_size]
            real_chunk = real_normed_items[start : start + batch_size]
            logits_fake = self.discriminator_train(self._augment_images(fake_chunk))
            if real_features is None:
                logits_real = self.discriminator_train(self._augment_images(real_chunk))
            else:
                logits_real = self.discriminator_train(
                    real_features[start : start + batch_size], from_features=True
                )
            real_per_image = gan_logits_per_image(logits_real).detach()
            fake_per_image = gan_logits_per_image(logits_fake).detach()
            chunk_loss = gan_discriminator_loss(
                logits_real, logits_fake, self.config.gan.discriminator_loss
            )
            (chunk_loss * (len(fake_chunk) / num_images)).backward()
            sums[0] += chunk_loss.detach().float() * len(fake_chunk)
            sums[1] += real_per_image.float().sum()
            sums[2] += fake_per_image.float().sum()
            sums[3] += (real_per_image > fake_per_image).float().sum()
        return tuple(total / num_images for total in sums)  # type: ignore[return-value]

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        # See the train_step note: NCCL pins this thread to the GPU-local
        # NUMA node at comm init. Reset before iter() spawns the dataloader
        # prefetch/pool threads so they inherit the full mask.
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        return super().batch_generator(data_iterable)

    def train_step(self, data_iterator: Iterator) -> None:
        # NCCL pins the calling thread to the GPU-local NUMA node when a
        # communicator initializes, and (in 2.29.x) does not restore the
        # original mask afterwards. On this host two GPUs share a NUMA node,
        # which confines those ranks and their dataloader threads to a few
        # cores and starves their data supply. Reset to the full mask each
        # step; late lazy comm inits can re-pin, and one syscall per step is
        # negligible.
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        now = time.perf_counter()
        if self._step_end_time is not None:
            self._phase_profiler.record("between", now - self._step_end_time)
        decoder = self.model_parts[0]
        gan = self.config.gan
        step = self.step - 1
        total_steps = max(1, self.config.training.steps)
        use_gan = (
            step >= gan.discriminator_start(total_steps)
            and gan.discriminator_weight > 0
        )
        train_discriminator = (
            step >= gan.discriminator_update_start(total_steps)
            and gan.discriminator_weight > 0
        )
        gan_weight = gan.discriminator_weight
        if use_gan and gan.discriminator_weight_ramp_steps > 0:
            # Soften the GAN onset: the discriminator has already converged
            # when the adversarial term starts, so a full-weight step-one
            # gradient spike can knock the decoder out of its basin.
            ramp_progress = (step - gan.discriminator_start(total_steps) + 1) / (
                gan.discriminator_weight_ramp_steps
            )
            gan_weight *= min(ramp_progress, 1.0)
        use_perceptual = step >= gan.perceptual_start_step and gan.perceptual_weight > 0
        use_fm = use_gan and gan.feature_matching_weight > 0
        num_microbatches = self.gradient_accumulation_steps
        images_batches: list[ImageBatch] = []
        cached_latents: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | float]
        ] = []

        self.optimizers.zero_grad(set_to_none=True)
        self.disc_optimizer.zero_grad(set_to_none=True)
        self.discriminator.eval()
        self.discriminator.set_head_requires_grad(False)
        reconstruction_metric = perceptual_metric = adversarial_metric = None
        adaptive_metric = None
        generator_logits_metric = None
        fm_metric = None
        non_padding_tokens = 0
        padding_capacity_tokens = 0
        num_images_per_step = 0
        microbatch_targets: list[list[torch.Tensor]] | None = None
        step_real_features: list[list[torch.Tensor]] | None = None
        step_aug_params: AugmentationParams | None = None
        if use_fm:
            # Feature matching needs the step's real supervision features
            # before the first decode, and the discriminator phase below
            # reuses them for the real pass. Fetch and encode every
            # microbatch first; the supervision sizes follow from grid_thw
            # (unpatchify maps each grid entry to (h * patch_size,
            # w * patch_size)) without decoding. The frozen encoder is
            # deterministic and consumes no RNG, so this hoist leaves the
            # random stream of the training loop unchanged.
            microbatch_targets = []
            with torch.no_grad():
                for _ in range(num_microbatches):
                    with self._phase_profiler.phase("data"):
                        images, encoder_input = self._next_images(data_iterator)
                    images_batches.append(images)
                    with self._phase_profiler.phase("encode"):
                        latents, grid_thw, fps, temporal_start = self._encode(
                            images, encoder_input
                        )
                    cached_latents.append((latents, grid_thw, fps, temporal_start))
                    batch_tokens = int(grid_thw.prod(dim=-1).sum().item())
                    self.metrics_processor.ntokens_since_last_log += batch_tokens
                    non_padding_tokens += batch_tokens
                    self._tokens_current_epoch += batch_tokens
                    padding_capacity_tokens += (
                        self._static_sequence_length
                        if self._static_sequence_length > 0
                        else batch_tokens
                    )
                    num_images_per_step += len(self._image_items(images))
                    target_sizes: list[tuple[int, int]] = [
                        (
                            int(entry[1]) * decoder.patch_size,
                            int(entry[2]) * decoder.patch_size,
                        )
                        for entry in grid_thw.reshape(-1, 3)
                    ]
                    microbatch_targets.append(
                        self._image_items(
                            self._supervision_images(images, target_sizes)
                        )
                    )
                flat_targets = [
                    target
                    for target_items in microbatch_targets
                    for target in target_items
                ]
                # One augmentation draw per step image, in the step's image
                # order. The same params augment the real-feature cache below
                # and the gen-phase fakes: the discriminator-phase real pass
                # keeps its augmented distribution (no augmentation-detecting
                # shortcut against the augmented fakes), and feature matching
                # compares spatially aligned real/fake pairs.
                step_aug_params = self.discriminator_augmentation.sample_params(
                    len(flat_targets), self.device
                )
                augmented_reals = self._augment_images(
                    [target * 2.0 - 1.0 for target in flat_targets],
                    params=step_aug_params,
                )
                step_real_features = self.discriminator.cache_real_features(
                    [(real + 1.0) * 0.5 for real in augmented_reals]
                )
        feature_offset = 0
        for microbatch_index in range(num_microbatches):
            if microbatch_targets is None:
                with self._phase_profiler.phase("data"):
                    images, encoder_input = self._next_images(data_iterator)
                images_batches.append(images)
                image_items = self._image_items(images)
                # Encode once per microbatch; the encoder is frozen and
                # deterministic, so the discriminator phase below re-decodes
                # these cached clean latents instead of re-running the vision
                # tower.
                with self._phase_profiler.phase("encode"):
                    latents, grid_thw, fps, temporal_start = self._encode(
                        images, encoder_input
                    )
                cached_latents.append((latents, grid_thw, fps, temporal_start))
                batch_tokens = int(grid_thw.prod(dim=-1).sum().item())
                self.metrics_processor.ntokens_since_last_log += batch_tokens
                target_items = None
                real_features_items = None
                aug_params_items = None
            else:
                images = images_batches[microbatch_index]
                image_items = self._image_items(images)
                latents, grid_thw, fps, temporal_start = cached_latents[
                    microbatch_index
                ]
                target_items = microbatch_targets[microbatch_index]
                assert step_real_features is not None and step_aug_params is not None
                feature_end = feature_offset + len(target_items)
                real_features_items = step_real_features[feature_offset:feature_end]
                aug_params_items = AugmentationParams(
                    gates=step_aug_params.gates,
                    uniforms=step_aug_params.uniforms[:, feature_offset:feature_end],
                )
                feature_offset = feature_end
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda"
                and self.config.training.dtype == "bfloat16",
            ):
                with self._phase_profiler.phase("decode"):
                    recon_items = self._decode(
                        decoder,
                        self._add_latent_noise(
                            latents, grid_thw, self.encoder.noise_tau
                        ),
                        grid_thw,
                        fps,
                        temporal_start,
                    )
                with self._phase_profiler.phase("supervision"):
                    if target_items is None:
                        target_sizes = [
                            (reconstruction.shape[-2], reconstruction.shape[-1])
                            for reconstruction in recon_items
                        ]
                        target_items = self._image_items(
                            self._supervision_images(images, target_sizes)
                        )
                    reconstruction_loss = torch.stack(
                        [
                            F.l1_loss(reconstruction, target)
                            for reconstruction, target in zip(
                                recon_items, target_items, strict=True
                            )
                        ]
                    ).mean()
                with self._phase_profiler.phase("perceptual"):
                    perceptual_loss = (
                        self._perceptual_loss(
                            [target * 2.0 - 1.0 for target in target_items],
                            [
                                reconstruction * 2.0 - 1.0
                                for reconstruction in recon_items
                            ],
                        )
                        if use_perceptual
                        else reconstruction_loss.new_zeros(())
                    )
                reconstruction_total = (
                    reconstruction_loss + gan.perceptual_weight * perceptual_loss
                )
                if use_gan:
                    with self._phase_profiler.phase("gan"):
                        fake_items = [
                            reconstruction * 2.0 - 1.0 for reconstruction in recon_items
                        ]
                        if use_fm:
                            assert aug_params_items is not None
                            # Each fake is augmented with the same draws as its
                            # paired real (whose augmented features are in the
                            # step-start cache), so the feature-matching pairs
                            # stay spatially aligned. The draws come from the
                            # same distribution as fresh sampling, so the
                            # adversarial loss keeps its marginal.
                            fake_augmented = self._augment_images(
                                fake_items, params=aug_params_items
                            )
                            logits_fake, fake_features = self.discriminator_train(
                                fake_augmented, return_features=True
                            )
                            assert real_features_items is not None
                            fm_loss = self.discriminator.feature_matching(
                                real_features_items, fake_features
                            )
                        else:
                            fake_augmented = self._augment_images(fake_items)
                            logits_fake = self.discriminator_train(fake_augmented)
                            fm_loss = None
                        generator_logits_metric = gan_logits_mean(logits_fake).detach()
                        adversarial_loss = gan_generator_loss(
                            logits_fake, gan.generator_loss
                        )
                        # Feature matching folds into the GAN term so the
                        # adaptive weight balances the whole GAN-side gradient
                        # against reconstruction.
                        adversarial_total = (
                            adversarial_loss
                            if fm_loss is None
                            else adversarial_loss
                            + gan.feature_matching_weight * fm_loss
                        )
                        with self._dmuon_reduce_suppressed():
                            adaptive_weight = self._adaptive_weight(
                                reconstruction_total,
                                adversarial_total,
                                decoder.decoder_pred.weight,
                                gan.max_adaptive_weight,
                            )
                        if adaptive_weight is None:
                            # A None gradient probe would silently disable the
                            # GAN term; fall back to zero and surface it once.
                            adaptive_weight = reconstruction_loss.new_zeros(())
                            if not self._warned_adaptive_weight_unavailable:
                                logger.warning(
                                    "RAE adaptive weight probe returned no "
                                    "gradient for decoder_pred.weight; the GAN "
                                    "term is scaled by 0 until this resolves"
                                )
                                self._warned_adaptive_weight_unavailable = True
                        total_loss = (
                            reconstruction_total
                            + gan_weight * adaptive_weight * adversarial_total
                        )
                else:
                    adversarial_loss = reconstruction_loss.new_zeros(())
                    adaptive_weight = reconstruction_loss.new_zeros(())
                    fm_loss = None
                    total_loss = reconstruction_total
                with self._phase_profiler.phase("backward"):
                    (total_loss / num_microbatches).backward()
            if microbatch_targets is None:
                non_padding_tokens += batch_tokens
                self._tokens_current_epoch += batch_tokens
                padding_capacity_tokens += (
                    self._static_sequence_length
                    if self._static_sequence_length > 0
                    else batch_tokens
                )
                num_images_per_step += len(image_items)
            reconstruction_metric = reconstruction_loss.detach()
            perceptual_metric = perceptual_loss.detach()
            adversarial_metric = adversarial_loss.detach()
            adaptive_metric = adaptive_weight.detach()
            if fm_loss is not None:
                fm_metric = fm_loss.detach()

        assert (
            reconstruction_metric is not None
            and perceptual_metric is not None
            and adversarial_metric is not None
            and adaptive_metric is not None
        )
        with self._phase_profiler.phase("optimizer"):
            decoder_grad_norm = self._decoder_grad_norm()
            self.optimizers.step()
            self.lr_schedulers.step()
            self._update_ema()

        image_items = [
            image for batch in images_batches for image in self._image_items(batch)
        ]
        metric_source = image_items[0]
        disc_loss = metric_source.new_zeros(())
        disc_grad_norm = metric_source.new_zeros(())
        discriminator_real_metric = discriminator_fake_metric = metric_source.new_zeros(
            ()
        )
        discriminator_accuracy_metric = metric_source.new_zeros(())
        dino_distance_metric = metric_source.new_zeros(())
        if train_discriminator:
            self._phase_profiler.start("disc")
            self.discriminator.set_head_requires_grad(True)
            self.discriminator_train.train()
            # RAEv2 decodes discriminator fakes with the generator in eval
            # mode; this also pins the residual-dropout masks off so the
            # discriminator sees deterministic decodes.
            decoder_was_training = decoder.training
            decoder.eval()
            for update_index in range(gan.discriminator_updates):
                self.disc_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda"
                    and self.config.training.dtype == "bfloat16",
                ):
                    fake_items = [
                        fake
                        for latents, grid_thw, fps, temporal_start in cached_latents
                        for fake in self._decode(
                            decoder,
                            latents,
                            grid_thw,
                            fps,
                            temporal_start,
                        )
                    ]
                target_sizes = [(fake.shape[-2], fake.shape[-1]) for fake in fake_items]
                real_items = self._image_items(
                    self._supervision_images(image_items, target_sizes)
                )
                fake_normed_items = [
                    (fake * 2.0 - 1.0).clamp(-1.0, 1.0) for fake in fake_items
                ]
                fake_normed_items = [
                    torch.round((fake + 1.0) * 127.5) / 127.5 - 1.0
                    for fake in fake_normed_items
                ]
                real_normed_items = [real * 2.0 - 1.0 for real in real_items]
                (
                    disc_loss,
                    discriminator_real_metric,
                    discriminator_fake_metric,
                    discriminator_accuracy_metric,
                ) = self._update_discriminator(
                    fake_normed_items,
                    real_normed_items,
                    real_features=step_real_features,
                )
                if update_index == 0:
                    # Logging-only DINO feature distance (uncalibrated
                    # LPIPS-style metric) on a 16-pair subsample. Eager and
                    # no-gradient; costs one small backbone forward per step.
                    with torch.no_grad(), torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda"
                        and self.config.training.dtype == "bfloat16",
                    ):
                        dino_distance_metric = self.discriminator.feature_distance(
                            real_normed_items[:16], fake_normed_items[:16]
                        )
                disc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.config.training.max_norm
                )
                self.disc_optimizer.step()
                self.disc_scheduler.step()
            decoder.train(decoder_was_training)
            self.discriminator.eval()
            self.discriminator.set_head_requires_grad(False)
            self._phase_profiler.stop("disc")

        phase_report = self._phase_profiler.collect()
        if phase_report:
            rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
            summary = " ".join(
                f"{name}={cpu:.2f}/{gpu:.2f}"
                for name, (cpu, gpu) in sorted(phase_report.items())
            )
            logger.info(
                f"[step {self.step} rank {rank}] phase cpu/gpu seconds: {summary}"
            )

        # The dataloader's epoch counter advances when the stream wraps, which
        # runs ahead of trainer consumption by the shuffle window and prefetch
        # buffer; the finished epoch's token count excludes that in-flight
        # tail, which is instead counted into the next epoch.
        epochs_completed = self._track_epoch_tokens()

        if self.metrics_processor.should_log(self.step):
            log_stage1_metrics(
                self.step,
                (
                    reconstruction_metric,
                    perceptual_metric,
                    adversarial_metric,
                    disc_loss,
                    adaptive_metric,
                    decoder_grad_norm,
                    disc_grad_norm,
                    generator_logits_metric
                    if generator_logits_metric is not None
                    else metric_source.new_zeros(()),
                    discriminator_real_metric,
                    discriminator_fake_metric,
                    discriminator_accuracy_metric,
                    dino_distance_metric,
                    fm_metric if fm_metric is not None else metric_source.new_zeros(()),
                ),
                metrics_processor=self.metrics_processor,
                non_padding_ratio=non_padding_tokens / padding_capacity_tokens,
                num_images_per_step=float(num_images_per_step),
                epoch=(
                    None if epochs_completed is None else float(self._last_seen_epoch)
                ),
                tokens_last_epoch=self._tokens_last_epoch,
            )
        self._step_end_time = time.perf_counter()

    def _track_epoch_tokens(self) -> int | None:
        """Fold consumed tokens into per-epoch totals at stream boundaries.

        Returns the dataloader's completed-epoch count, or None when the
        dataloader does not track epochs.
        """
        epochs_completed = getattr(self.dataloader, "epochs_completed", None)
        if epochs_completed is not None and epochs_completed > self._last_seen_epoch:
            self._tokens_last_epoch = self._tokens_current_epoch
            self._tokens_current_epoch = 0
            self._last_seen_epoch = epochs_completed
        return epochs_completed

    def should_continue_training(self) -> bool:
        if not super().should_continue_training():
            return False
        if self.config.epochs is None:
            return True
        epochs_completed = getattr(self.dataloader, "epochs_completed", None)
        if epochs_completed is None:
            if not self._warned_epoch_tracking_missing:
                logger.warning(
                    "epochs=%d requested but the dataloader does not track "
                    "epoch completion; the epochs stop is disabled",
                    self.config.epochs,
                )
                self._warned_epoch_tracking_missing = True
            return True
        # Every rank evaluates this hook once per loop iteration, so the
        # all-reduce stays synchronized and all ranks stop at the same step.
        if (
            dist.is_available()
            and dist.is_initialized()
            and self.parallel_dims.dp_enabled
        ):
            batch_mesh = self.parallel_dims.get_mesh("batch")
            if batch_mesh.size() > 1:
                completed = torch.tensor(
                    epochs_completed, device=self.device, dtype=torch.long
                )
                dist.all_reduce(
                    completed, op=dist.ReduceOp.MIN, group=batch_mesh.get_group()
                )
                epochs_completed = int(completed.item())
        return epochs_completed < self.config.epochs

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["rae_tokens_current_epoch"] = self._tokens_current_epoch
        state["rae_last_seen_epoch"] = self._last_seen_epoch
        state["rae_tokens_last_epoch"] = self._tokens_last_epoch
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._tokens_current_epoch = state_dict.get("rae_tokens_current_epoch", 0)
        self._last_seen_epoch = state_dict.get("rae_last_seen_epoch", 0)
        self._tokens_last_epoch = state_dict.get("rae_tokens_last_epoch")

    @contextmanager
    def _dmuon_reduce_suppressed(self) -> Generator[None]:
        """Inert dmuon backward reduce hooks while autograd.grad probes run.

        The adaptive-weight probe executes partial backward passes outside
        the once-per-forward protocol dmuon's post-backward hooks assume.
        """
        decoder = self.model_parts[0]
        if getattr(decoder, "_dmuon_enabled", False) and hasattr(
            decoder, "_dedicated_comm_ctx"
        ):
            dmuon = load_dmuon()
            with dmuon.suppress_grad_reduce(decoder):
                yield
        else:
            yield

    @staticmethod
    def _adaptive_weight(
        reconstruction_loss: torch.Tensor,
        adversarial_loss: torch.Tensor,
        layer: torch.Tensor,
        max_weight: float,
    ) -> torch.Tensor | None:
        recon_grad = torch.autograd.grad(
            reconstruction_loss, layer, retain_graph=True, allow_unused=True
        )[0]
        gan_grad = torch.autograd.grad(
            adversarial_loss, layer, retain_graph=True, allow_unused=True
        )[0]
        if recon_grad is None or gan_grad is None:
            return None
        return (
            (
                torch.linalg.vector_norm(recon_grad)
                / (torch.linalg.vector_norm(gan_grad) + 1e-6)
            )
            .clamp(0, max_weight)
            .detach()
        )


def _nearest_neighbor_indices(images: Sequence[torch.Tensor]) -> torch.Tensor:
    """Per-anchor nearest neighbor in 32x32 grayscale pixel space (L2)."""
    gray_ND = torch.stack(
        [
            F.interpolate(
                image_CHW.unsqueeze(0).float(),
                size=(32, 32),
                mode="bilinear",
                antialias=True,
            )
            .mean(dim=1)
            .flatten(1)[0]
            for image_CHW in images
        ]
    )
    distances_NN = torch.cdist(gray_ND, gray_ND)
    distances_NN.fill_diagonal_(float("inf"))
    return distances_NN.argmin(dim=1)


def _r_phi_ratio(
    phi_a: Sequence[torch.Tensor],
    phi_b: Sequence[torch.Tensor],
    phi_m: Sequence[torch.Tensor],
) -> float | None:
    """Mean over pairs of 2*|phi(m)-phi(a)| / |phi(b)-phi(a)| (L1 means).

    m is the (quantized) midpoint of a and b. Pairs whose endpoints coincide
    in feature space (zero denominator) are skipped; None when no pair
    qualifies.
    """
    ratios = []
    for anchor, neighbor, midpoint in zip(phi_a, phi_b, phi_m, strict=True):
        denominator = (neighbor.float() - anchor.float()).abs().mean()
        if denominator.item() == 0.0:
            continue
        numerator = 2.0 * (midpoint.float() - anchor.float()).abs().mean()
        ratios.append((numerator / denominator).item())
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


class RAEValidator(BaseValidator):
    """Validate RAE reconstructions and optionally log a comparison image."""

    def __init__(self, config: BaseValidator.Config, trainer: RAEStage1Trainer) -> None:
        super().__init__(config=config)
        self.trainer = trainer

    @staticmethod
    def _comparison_image(
        target_CHW: torch.Tensor,
        reconstruction_CHW: torch.Tensor,
        *,
        step: int,
    ) -> Any:
        import wandb

        comparison_CHW = torch.cat(
            [target_CHW.float().clamp(0, 1), reconstruction_CHW.float().clamp(0, 1)],
            dim=-1,
        )
        comparison_HWC = (
            comparison_CHW.mul(255.0)
            .round()
            .to(dtype=torch.uint8)
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        return wandb.Image(
            comparison_HWC,
            caption=f"step {step}: ground truth | reconstruction",
        )

    def _validation_dataloader(self) -> BaseDataLoader:
        config = self.config
        dataloader_config = getattr(config, "dataloader", None)
        if dataloader_config is None:
            dataloader_config = self.trainer.config.dataloader
        else:
            # The base Trainer validator defaults to a text C4 loader. RAE
            # validation must use the configured Qwen media loader instead.
            collator = getattr(dataloader_config, "collator", None)
            if not isinstance(collator, RAEQwenCollator.Config):
                dataloader_config = self.trainer.config.dataloader
        dataloader_config = replace(
            dataloader_config,
            repeat=self.config.steps != -1,
            shuffle=False,
        )
        parallel_dims = self.trainer.parallel_dims
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            dp_world_size = batch_mesh.size()
            dp_rank = batch_mesh.get_local_rank()
        else:
            dp_world_size = 1
            dp_rank = 0
        return dataloader_config.build(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=self.trainer.tokenizer,
            max_context_length=self.trainer.config.training.max_context_length,
            num_tokens_per_batch=self.trainer.config.training.num_tokens_per_microbatch_per_dp_rank,
        )

    @torch.no_grad()
    def _off_manifold_metrics(
        self,
        eval_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, float]:
        """R_phi off-manifold penalties and a normalized feature distance.

        Anchors pair with their nearest neighbor within the same resolution
        (feature diffs require a shared token grid). For each pair (a, b) the
        midpoint m = (a+b)/2 in [-1, 1] is quantized to the 8-bit grid used
        for discriminator fakes, and R_phi = 2*|phi(m)-phi(a)| / |phi(b)-phi(a)|
        is computed per probed backbone depth and on the concatenated head
        logits. dino_dist_normalized divides the real-vs-reconstruction
        backbone distance by the mean real-to-neighbor distance. Everything
        is eager and no-grad, once per validation round.
        """
        discriminator = self.trainer.discriminator
        was_training = discriminator.training
        discriminator.eval()
        try:
            targets = [target.float() for target, _ in eval_pairs]
            groups: dict[tuple[int, int], list[int]] = {}
            for index, target in enumerate(targets):
                groups.setdefault((target.shape[-2], target.shape[-1]), []).append(
                    index
                )
            anchor_indices: list[int] = []
            neighbor_indices: list[int] = []
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                neighbors = _nearest_neighbor_indices(
                    [targets[index] for index in indices]
                )
                anchor_indices.extend(indices)
                neighbor_indices.extend(
                    indices[int(neighbor)] for neighbor in neighbors.tolist()
                )
            if not anchor_indices:
                return {}
            a01 = [targets[index] for index in anchor_indices]
            b01 = [targets[index] for index in neighbor_indices]
            mid_neg1 = [
                torch.round((ta + tb) * 127.5) / 127.5 - 1.0
                for ta, tb in zip(a01, b01, strict=True)
            ]
            mid01 = [(midpoint + 1.0) * 0.5 for midpoint in mid_neg1]
            num_pairs = len(a01)
            features = discriminator.features(a01 + b01 + mid01)
            phi_a, phi_b, phi_m = (
                features[:num_pairs],
                features[num_pairs : 2 * num_pairs],
                features[2 * num_pairs :],
            )
            metrics: dict[str, float] = {}
            neighbor_distances: list[float] = []
            depth_ratios: list[float] = []
            for depth in range(len(phi_a[0])):
                anchors = [per_image[depth] for per_image in phi_a]
                neighbors = [per_image[depth] for per_image in phi_b]
                midpoints = [per_image[depth] for per_image in phi_m]
                ratio = _r_phi_ratio(anchors, neighbors, midpoints)
                if ratio is not None:
                    metrics[f"rae/r_phi_backbone_depth{depth}"] = ratio
                    depth_ratios.append(ratio)
                neighbor_distances.extend(
                    (neighbor.float() - anchor.float()).abs().mean().item()
                    for anchor, neighbor in zip(anchors, neighbors, strict=True)
                )
            if depth_ratios:
                metrics["rae/r_phi_backbone"] = sum(depth_ratios) / len(depth_ratios)
            # Disc-feature R_phi: phi is the concatenated per-patch logits of
            # all heads, the learned adversarial space.
            logits = discriminator(
                [image * 2.0 - 1.0 for image in a01]
                + [image * 2.0 - 1.0 for image in b01]
                + mid_neg1
            )
            disc_a, disc_b, disc_m = (
                logits[:num_pairs],
                logits[num_pairs : 2 * num_pairs],
                logits[2 * num_pairs :],
            )
            disc_ratio = _r_phi_ratio(
                [logit.flatten() for logit in disc_a],
                [logit.flatten() for logit in disc_b],
                [logit.flatten() for logit in disc_m],
            )
            if disc_ratio is not None:
                metrics["rae/r_phi_disc"] = disc_ratio
            dino_distance = discriminator.feature_distance(
                [(target * 2.0 - 1.0) for target, _ in eval_pairs[:16]],
                [
                    (reconstruction * 2.0 - 1.0).clamp(-1.0, 1.0)
                    for _, reconstruction in eval_pairs[:16]
                ],
            )
            mean_neighbor_distance = sum(neighbor_distances) / len(neighbor_distances)
            if mean_neighbor_distance > 0:
                metrics["rae/dino_dist_normalized"] = (
                    float(dino_distance) / mean_neighbor_distance
                )
            return metrics
        finally:
            discriminator.train(was_training)

    @torch.no_grad()
    def validate(self, model_parts: list[torch.nn.Module], step: int) -> None:
        decoder = model_parts[0]
        was_training = decoder.training
        decoder.eval()
        collect_r_phi = self.trainer.config.discriminator.backbone_kind == "hf"
        validation_dataloader = self._validation_dataloader()
        validation_iterator = iter(iterate_and_close_dataloader(validation_dataloader))
        losses: list[torch.Tensor] = []
        eval_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        comparison_image = None
        num_batches = 0
        num_tokens = 0
        num_images = 0
        padding_capacity_tokens = 0
        try:
            while self.config.steps == -1 or num_batches < self.config.steps:
                try:
                    images, encoder_input = self.trainer._next_images(
                        validation_iterator,
                        count_training_stats=False,
                    )
                except StopIteration:
                    break
                with (
                    self.trainer.train_context(),
                    torch.autocast(
                        device_type=self.trainer.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.trainer.device.type == "cuda"
                        and self.trainer.config.training.dtype == "bfloat16",
                    ),
                ):
                    reconstructions = self.trainer._encode_decode(
                        decoder,
                        images,
                        encoder_input,
                        add_noise=False,
                    )
                grid_thw = self.trainer.encoder.last_grid_thw
                if grid_thw is None:
                    raise RuntimeError(
                        "RAE validation encoder did not return grid metadata"
                    )
                batch_tokens = int(grid_thw.prod(dim=-1).sum().item())
                num_tokens += batch_tokens
                num_images += len(self.trainer._image_items(images))
                padding_capacity_tokens += (
                    self.trainer._static_sequence_length
                    if self.trainer._static_sequence_length > 0
                    else batch_tokens
                )
                self.trainer.metrics_processor.ntokens_since_last_log += batch_tokens
                target_sizes = [
                    (reconstruction.shape[-2], reconstruction.shape[-1])
                    for reconstruction in reconstructions
                ]
                targets = self.trainer._image_items(
                    self.trainer._supervision_images(images, target_sizes)
                )
                losses.extend(
                    F.l1_loss(reconstruction, target).detach()
                    for reconstruction, target in zip(
                        reconstructions, targets, strict=True
                    )
                )
                if collect_r_phi and len(eval_pairs) < 32:
                    eval_pairs.extend(
                        (target.detach(), reconstruction.detach())
                        for target, reconstruction in zip(
                            targets, reconstructions, strict=True
                        )
                    )
                if comparison_image is None and (
                    self.trainer.config.metrics.enable_wandb
                    or self.trainer.config.metrics.enable_swanlab
                ):
                    comparison_image = self._comparison_image(
                        targets[0],
                        reconstructions[0],
                        step=step,
                    )
                num_batches += 1
        finally:
            decoder.train(was_training)

        if not losses:
            raise RuntimeError("RAE validation dataloader produced no media batches")
        extras: dict[str, Any] = {
            "validation_metrics/num_tokens": num_tokens,
            "validation_metrics/num_batches": num_batches,
            "validation_metrics/non_padding_ratio": (
                num_tokens / padding_capacity_tokens
            ),
            "validation_metrics/num_images_per_step": num_images,
        }
        if collect_r_phi and len(eval_pairs) >= 2:
            extras.update(self._off_manifold_metrics(eval_pairs[:32]))
        if comparison_image is not None:
            extras[
                "validation_images/ground_truth_vs_reconstruction"
            ] = comparison_image
        self.trainer.metrics_processor.log_validation(
            loss=float(torch.stack(losses).mean().item()),
            step=step,
            extra_metrics=extras,
        )


__all__ = [
    "DiscriminatorAugmentation",
    "RAEGANAugmentConfig",
    "RAEGANConfig",
    "RAEStage1Trainer",
    "RAEValidator",
    "log_stage1_metrics",
]
