# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import math
import os
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.parallel import DistributedDataParallel

from torchtitan.components.checkpointer import LR_SCHEDULER, MODEL, OPTIMIZER
from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.optimizer.dmuon import load_dmuon
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer
from ..decoder import create_rae_static_varlen_metadata
from ..discriminator import (
    gan_discriminator_loss,
    gan_generator_loss,
    RAEFeatureDiscriminator,
    RAEPerceptualLoss,
)
from ..encoder import FrozenRAEEncoder, RAEEncoderConfig

from .augmentation import DiscriminatorAugmentation
from .graphs import RAEDiscriminatorGraph
from .metrics import log_stage1_metrics
from .validation import RAEValidator


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
    perceptual_weight: float = 1.0
    discriminator_updates: int = 1
    discriminator_chunk_size: int = 128
    """Fixed image count per discriminator update chunk. The CUDA graph is
    captured once at this shape, so the update's private memory pool is bounded
    independently of how many images a packed step contains."""
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
        if self.discriminator_chunk_size <= 0:
            raise ValueError("gan.discriminator_chunk_size must be positive")
        if self.discriminator_weight < 0 or self.perceptual_weight < 0:
            raise ValueError("GAN and perceptual weights must be non-negative")
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
        if self.perceptual_kind not in {"fixed", "lpips"}:
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
        ema: nn.Module | None,
        ema_shadow: dict[str, torch.Tensor] | None,
    ):
        self.decoder = decoder
        self.discriminator = discriminator
        self.ema = ema
        self.ema_shadow = ema_shadow

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
        ema_state = (
            self.ema.state_dict()
            if self.ema is not None
            else {name: value for name, value in (self.ema_shadow or {}).items()}
        )
        return {
            "decoder": decoder_state,
            "discriminator": self.discriminator.state_dict(),
            "ema": ema_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        decoder_state = state_dict.get("decoder", state_dict)
        if getattr(self.decoder, "_dmuon_enabled", False) and hasattr(
            self.decoder, "_dedicated_comm_ctx"
        ):
            dmuon = load_dmuon()

            dmuon.set_model_state_dict(self.decoder, decoder_state)
            for name, buffer in self.decoder.named_buffers():
                if name in decoder_state:
                    buffer.copy_(
                        decoder_state[name].to(device=buffer.device, dtype=buffer.dtype)
                    )
        else:
            self.decoder.load_state_dict(decoder_state, strict=False)
        if "discriminator" in state_dict:
            self.discriminator.load_state_dict(
                state_dict["discriminator"], strict=False
            )
        if "ema" in state_dict:
            if self.ema is not None:
                self.ema.load_state_dict(state_dict["ema"], strict=False)
            elif self.ema_shadow is not None:
                for name, value in state_dict["ema"].items():
                    if name in self.ema_shadow:
                        self.ema_shadow[name].copy_(value)


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
        self._adaptive_weight_enabled = not any(
            hasattr(optimizer, "_dedicated_params") for optimizer in self.optimizers
        )
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
        self._cuda_graphs_enabled = (
            not config.training.disable_cuda_graphs and self.device.type == "cuda"
        )
        if config.compile.enable and "discriminator" in config.compile.components:
            self.discriminator.compile_forward(backend=config.compile.backend)
        if (
            dist.is_available()
            and dist.is_initialized()
            and self.parallel_dims.dp_enabled
            and not self._cuda_graphs_enabled
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
        if (
            config.compile.enable
            and "discriminator" in config.compile.components
            and config.gan.discriminator_weight > 0
        ):
            canvas = self.discriminator.input_canvas_size
            if canvas is not None:
                # Warm the generator-side compiled path during initialization
                # (covered by the init timeout): the first GAN step would
                # otherwise compile mid-run, and slow per-rank compilation can
                # stall the DP collectives. The warmup must replicate the
                # runtime guards (eval mode, frozen head, bf16 autocast,
                # grad-enabled chunk of gan.discriminator_chunk_size rows) or
                # dynamo compiles a separate variant and the warmup is wasted.
                autocast_bf16 = (
                    self.device.type == "cuda" and config.training.dtype == "bfloat16"
                )
                dummy_BCHW = torch.zeros(
                    config.gan.discriminator_chunk_size,
                    3,
                    canvas,
                    canvas,
                    device=self.device,
                    dtype=torch.bfloat16 if autocast_bf16 else torch.float32,
                    requires_grad=True,
                )
                self.discriminator.eval()
                self.discriminator.set_head_requires_grad(False)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_bf16,
                ):
                    self.discriminator(self.discriminator_augmentation(dummy_BCHW))
                self.discriminator.set_head_requires_grad(True)
        self._discriminator_graphs: dict[tuple[int, ...], RAEDiscriminatorGraph] = {}
        self._graph_failures: set[tuple[str, tuple[Any, ...]]] = set()
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
        if self._cuda_graphs_enabled:
            canvas = self.discriminator.input_canvas_size
            if canvas is not None:
                self._precapture_discriminator_graph(canvas)
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
        self.ema_model, self.ema_shadow = self._build_ema(decoder)

        if getattr(self.checkpointer, "enable", False):
            self.checkpointer.states[MODEL] = _RAEModelState(
                decoder,
                self.discriminator,
                self.ema_model,
                self.ema_shadow,
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

    def _build_ema(
        self, decoder: nn.Module
    ) -> tuple[nn.Module, dict[str, torch.Tensor] | None]:
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
        missing, _unexpected = ema_model.load_state_dict(normalized_state, strict=False)
        if missing:
            raise RuntimeError(
                "RAE EMA initialization is missing decoder parameters: "
                + ", ".join(missing)
            )
        ema_model.eval()
        ema_model.requires_grad_(False)
        return ema_model, None

    @torch.no_grad()
    def _update_ema(self) -> None:
        decoder = self.model_parts[0]
        decay = self.config.gan.ema_decay
        if self.ema_model is not None:
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
            return
        raise RuntimeError("RAE EMA shadow state is no longer supported")

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
                if torch.is_tensor(value) or name == "media_kind"
            }
        if isinstance(images, torch.Tensor):
            if images.ndim == 5:
                if images.shape[1] != 1:
                    raise ValueError(
                        "RAE Stage 1 image training accepts only one-frame media"
                    )
                images = images[:, 0]
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
                    [(target_size,) * 2 for _ in image_items]
                    if target_size is not None
                    else [tuple(image.shape[-2:]) for image in image_items]
                )
            if len(target_sizes) != len(image_items):
                raise ValueError("Supervision target sizes must match image count")
            return [
                self._resize_supervision_image(image, size)
                for image, size in zip(image_items, target_sizes, strict=True)
            ]
        if images_BCHW.ndim == 5:
            batch_size, num_frames, channels, height, width = images_BCHW.shape
            flattened = images_BCHW.reshape(
                batch_size * num_frames, channels, height, width
            )
            supervised = self._supervision_images(flattened)
            return supervised.reshape(
                batch_size,
                num_frames,
                channels,
                supervised.shape[-2],
                supervised.shape[-1],
            )
        if images_BCHW.ndim != 4:
            raise ValueError("RAE supervision expects BCHW or BTCHW images")
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
        """Run the frozen encoder once and return clean latents with metadata."""
        encoder_source = encoder_input if encoder_input is not None else images
        encoded = self.encoder(
            encoder_source,
            add_noise=False,
            return_grid_thw=True,
        )
        if not isinstance(encoded, tuple):
            raise RuntimeError("RAE encoder must return grid metadata for Stage 1")
        latents, grid_thw = encoded
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
        # One grouped call per microbatch: equal-shape images share backbone
        # forwards, and the real branch runs under no_grad.
        return self.perceptual_loss.forward_per_sample_list(
            real_items, fake_items
        ).mean()

    def _stack_canvas_images(self, images: list[torch.Tensor]) -> torch.Tensor | None:
        """Letterbox [-1, 1] CHW images onto the discriminator's square canvas.

        A fixed (B, 3, canvas, canvas) stack lets the compiled backbone and the
        CUDA-graph discriminator path run every microbatch regardless of the
        native resolutions in the batch. Returns None for backbones without a
        fixed canvas.
        """
        canvas = self.discriminator.input_canvas_size
        if canvas is None or not images:
            return None
        if any(image.ndim != 3 or image.shape[0] != 3 for image in images):
            return None
        patch = self.discriminator.canvas_patch_size
        letterboxed = []
        for image in images:
            height, width = image.shape[-2:]
            if (height, width) == (canvas, canvas):
                letterboxed.append(image)
                continue
            scale = min(canvas / height, canvas / width)
            target_height = max(
                patch, min(canvas, int(height * scale) // patch * patch)
            )
            target_width = max(patch, min(canvas, int(width * scale) // patch * patch))
            resized = F.interpolate(
                image.unsqueeze(0),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            canvas_image = image.new_zeros(3, canvas, canvas)
            top = (canvas - target_height) // 2
            left = (canvas - target_width) // 2
            canvas_image[
                :, top : top + target_height, left : left + target_width
            ] = resized
            letterboxed.append(canvas_image)
        return torch.stack(letterboxed)

    def _discriminator_generator_logits(
        self, images_BCHW: torch.Tensor
    ) -> torch.Tensor:
        """Generator-side discriminator logits over fixed-shape chunks.

        The compiled backbone re-specializes on batch size, and a mid-training
        recompile stalls one rank for tens of seconds while its DP peers wait
        in the gradient collective. Chunking to gan.discriminator_chunk_size
        keeps a single compiled shape, matching the discriminator-update path.
        Padded rows are sliced off the concatenated logits.
        """
        chunk_size = self.config.gan.discriminator_chunk_size
        num_images = images_BCHW.shape[0]
        logit_chunks = []
        for start in range(0, num_images, chunk_size):
            valid = min(chunk_size, num_images - start)
            chunk, _ = self._pad_to_bucket(
                images_BCHW[start : start + chunk_size], chunk_size
            )
            logits = self.discriminator_train(self.discriminator_augmentation(chunk))
            logit_chunks.append(logits[:valid])
        return torch.cat(logit_chunks)

    @staticmethod
    def _pad_to_bucket(
        images_BCHW: torch.Tensor, bucket: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a stacked batch up to a bucket multiple with zero images."""
        batch = images_BCHW.shape[0]
        target = ((batch + bucket - 1) // bucket) * bucket
        mask = images_BCHW.new_zeros(target, dtype=torch.float32)
        mask[:batch] = 1.0
        if target == batch:
            return images_BCHW, mask
        padding = images_BCHW.new_zeros(target - batch, *images_BCHW.shape[1:])
        return torch.cat([images_BCHW, padding], dim=0), mask

    def _get_discriminator_graph(
        self, shape: tuple[int, ...]
    ) -> RAEDiscriminatorGraph | None:
        if not self._cuda_graphs_enabled:
            return None
        if ("discriminator", shape) in self._graph_failures:
            return None
        graph = self._discriminator_graphs.get(shape)
        if graph is None:
            if len(self._discriminator_graphs) >= 32:
                # Bound graph capture cost when bucket shapes proliferate.
                return None
            graph = RAEDiscriminatorGraph(
                self.discriminator,
                self.discriminator_augmentation,
                discriminator_loss=self.config.gan.discriminator_loss,
                autocast_dtype=(
                    torch.bfloat16 if self.config.training.dtype == "bfloat16" else None
                ),
            )
            self._discriminator_graphs[shape] = graph
        return graph

    def _precapture_discriminator_graph(self, canvas: int) -> None:
        """Capture the discriminator CUDA graph at init with dummy inputs.

        Allocates the GAN-phase graph pool up front: a capture-time OOM fails
        at startup instead of mid-run at the GAN phase boundary, and the phase
        transition does not stall collectives on capture. Dummy dtypes mirror
        the runtime pipeline (bf16 canvas stacks under bf16 training).
        """
        chunk_size = self.config.gan.discriminator_chunk_size
        dtype = (
            torch.bfloat16
            if self.config.training.dtype == "bfloat16"
            else torch.float32
        )
        dummy_BCHW = torch.zeros(
            chunk_size, 3, canvas, canvas, device=self.device, dtype=dtype
        )
        result = self._run_discriminator_graph(dummy_BCHW, dummy_BCHW)
        self._zero_discriminator_gradients()
        if result is None:
            logger.warning(
                "RAE discriminator graph pre-capture failed; the GAN phase "
                "will fall back to the eager discriminator update"
            )

    def _run_discriminator_graph(
        self,
        fake_BCHW: torch.Tensor,
        real_BCHW: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Chunked discriminator update through one fixed-shape CUDA graph.

        Every chunk is padded to ``gan.discriminator_chunk_size`` and replayed
        through the same graph, so a step with ~1000 images holds a single
        private memory pool instead of one giant (or many per-batch) captures.
        Gradients accumulate as masked sums across chunks and are divided by
        the total valid count once, matching the eager path's mean reduction.
        Returns normalized (loss, real_logits, fake_logits, accuracy), or None
        when capture/replay fails (the caller then falls back to eager).
        """
        chunk_size = self.config.gan.discriminator_chunk_size
        chunk_shape = (chunk_size, *fake_BCHW.shape[1:])
        if ("discriminator", chunk_shape) in self._graph_failures:
            return None
        graph = self._get_discriminator_graph(chunk_shape)
        if graph is None:
            return None
        sums = [
            torch.zeros((), device=fake_BCHW.device, dtype=torch.float32)
            for _ in range(4)
        ]
        count = torch.zeros((), device=fake_BCHW.device, dtype=torch.float32)
        try:
            for start in range(0, fake_BCHW.shape[0], chunk_size):
                fake_chunk, valid_mask_B = self._pad_to_bucket(
                    fake_BCHW[start : start + chunk_size], chunk_size
                )
                real_chunk, _ = self._pad_to_bucket(
                    real_BCHW[start : start + chunk_size], chunk_size
                )
                output = graph(fake_chunk, real_chunk, valid_mask_B)
                sums[0] += output.loss_sum
                sums[1] += output.real_logits_sum
                sums[2] += output.fake_logits_sum
                sums[3] += output.accuracy_sum
                count += output.valid_count
        except Exception as error:
            # A failed chunk may have left partially accumulated gradients;
            # drop them so the eager fallback starts from a clean state.
            self._graph_failures.add(("discriminator", chunk_shape))
            self._zero_discriminator_gradients()
            logger.warning(
                "RAE discriminator CUDA graph unavailable for shape %s; "
                "falling back to eager loss (%s)",
                chunk_shape,
                error,
            )
            return None
        normalizer = count.clamp_min(1.0)
        for parameter in self.discriminator.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(normalizer)
        return tuple(total / normalizer for total in sums)  # type: ignore[return-value]

    def _update_discriminator_fixed_eager(
        self,
        fake_BCHW: torch.Tensor,
        real_BCHW: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager chunked discriminator update over fixed-canvas images.

        Each chunk's mean loss is weighted by its image count, so accumulated
        gradients match a single full-batch masked mean while peak memory stays
        bounded by the chunk size.
        """
        chunk_size = self.config.gan.discriminator_chunk_size
        num_images = fake_BCHW.shape[0]
        loss_sum = torch.zeros((), device=fake_BCHW.device, dtype=torch.float32)
        real_sum = torch.zeros((), device=fake_BCHW.device, dtype=torch.float32)
        fake_sum = torch.zeros((), device=fake_BCHW.device, dtype=torch.float32)
        accuracy_sum = torch.zeros((), device=fake_BCHW.device, dtype=torch.float32)
        for start in range(0, num_images, chunk_size):
            fake_chunk = fake_BCHW[start : start + chunk_size]
            real_chunk = real_BCHW[start : start + chunk_size]
            num_chunk = fake_chunk.shape[0]
            logits_fake = self.discriminator_train(
                self.discriminator_augmentation(fake_chunk)
            )
            logits_real = self.discriminator_train(
                self.discriminator_augmentation(real_chunk)
            )
            chunk_loss = gan_discriminator_loss(
                logits_real, logits_fake, self.config.gan.discriminator_loss
            )
            (chunk_loss * (num_chunk / num_images)).backward()
            loss_sum += chunk_loss.detach() * num_chunk
            real_sum += logits_real.mean(dim=-1).detach().sum()
            fake_sum += logits_fake.mean(dim=-1).detach().sum()
            accuracy_sum += (
                (logits_real.mean(dim=-1) > logits_fake.mean(dim=-1))
                .float()
                .detach()
                .sum()
            )
        return (
            loss_sum / num_images,
            real_sum / num_images,
            fake_sum / num_images,
            accuracy_sum / num_images,
        )

    def _zero_discriminator_gradients(self) -> None:
        if self._discriminator_graphs:
            for parameter in self.discriminator.parameters():
                if parameter.grad is not None:
                    parameter.grad.zero_()
        else:
            self.disc_optimizer.zero_grad(set_to_none=True)

    def _sync_discriminator_gradients(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        if not self.parallel_dims.dp_enabled:
            return
        batch_mesh = self.parallel_dims.get_mesh("batch")
        if batch_mesh.size() == 1:
            return
        group = batch_mesh.get_group()
        for parameter in self.discriminator.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, group=group)
                parameter.grad.div_(batch_mesh.size())

    def _augment_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        grouped: dict[tuple[int, int], list[int]] = {}
        for index, image in enumerate(images):
            grouped.setdefault(tuple(image.shape[-2:]), []).append(index)
        augmented = list(images)
        for indices in grouped.values():
            group = torch.stack([images[index] for index in indices])
            group = self.discriminator_augmentation(group)
            for group_index, image_index in enumerate(indices):
                augmented[image_index] = group[group_index]
        return augmented

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
        use_perceptual = step >= gan.perceptual_start_step and gan.perceptual_weight > 0
        num_microbatches = self.gradient_accumulation_steps
        images_batches: list[ImageBatch] = []
        cached_latents: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | float]
        ] = []

        self.optimizers.zero_grad(set_to_none=True)
        self._zero_discriminator_gradients()
        self.discriminator.eval()
        self.discriminator.set_head_requires_grad(False)
        reconstruction_metric = perceptual_metric = adversarial_metric = None
        adaptive_metric = None
        generator_logits_metric = None
        non_padding_tokens = 0
        padding_capacity_tokens = 0
        num_images_per_step = 0
        for _ in range(num_microbatches):
            with self._phase_profiler.phase("data"):
                images, encoder_input = self._next_images(data_iterator)
            images_batches.append(images)
            image_items = self._image_items(images)
            # Encode once per microbatch; the encoder is frozen and
            # deterministic, so the discriminator phase below re-decodes these
            # cached clean latents instead of re-running the vision tower.
            with self._phase_profiler.phase("encode"):
                latents, grid_thw, fps, temporal_start = self._encode(
                    images, encoder_input
                )
            cached_latents.append((latents, grid_thw, fps, temporal_start))
            batch_tokens = int(grid_thw.prod(dim=-1).sum().item())
            self.metrics_processor.ntokens_since_last_log += batch_tokens
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
                    target_sizes = [
                        tuple(reconstruction.shape[-2:])
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
                        fake_canvas_BCHW = self._stack_canvas_images(
                            [
                                reconstruction * 2.0 - 1.0
                                for reconstruction in recon_items
                            ]
                        )
                        if fake_canvas_BCHW is not None:
                            logits_fake = self._discriminator_generator_logits(
                                fake_canvas_BCHW
                            )
                        else:
                            fake_augmented = self._augment_images(
                                [
                                    reconstruction * 2.0 - 1.0
                                    for reconstruction in recon_items
                                ]
                            )
                            logits_fake = self.discriminator_train(fake_augmented)
                        generator_logits_metric = logits_fake.detach().mean()
                        adversarial_loss = gan_generator_loss(
                            logits_fake, gan.generator_loss
                        )
                        adaptive_weight = (
                            self._adaptive_weight(
                                reconstruction_total,
                                adversarial_loss,
                                decoder.decoder_pred.weight,
                                gan.max_adaptive_weight,
                            )
                            if self._adaptive_weight_enabled
                            else reconstruction_loss.new_ones(())
                        )
                        total_loss = (
                            reconstruction_total
                            + gan.discriminator_weight
                            * adaptive_weight
                            * adversarial_loss
                        )
                else:
                    adversarial_loss = reconstruction_loss.new_zeros(())
                    adaptive_weight = reconstruction_loss.new_zeros(())
                    total_loss = reconstruction_total
                with self._phase_profiler.phase("backward"):
                    (total_loss / num_microbatches).backward()
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
        if train_discriminator:
            self._phase_profiler.start("disc")
            self.discriminator.set_head_requires_grad(True)
            self.discriminator_train.train()
            # RAEv2 decodes discriminator fakes with the generator in eval
            # mode; this also pins the residual-dropout masks off so the
            # discriminator sees deterministic decodes.
            decoder_was_training = decoder.training
            decoder.eval()
            for _ in range(gan.discriminator_updates):
                self._zero_discriminator_gradients()
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
                target_sizes = [tuple(fake.shape[-2:]) for fake in fake_items]
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
                fake_BCHW = self._stack_canvas_images(fake_normed_items)
                real_BCHW = self._stack_canvas_images(real_normed_items)
                if fake_BCHW is not None and real_BCHW is not None:
                    fixed_output = None
                    if self._cuda_graphs_enabled:
                        fixed_output = self._run_discriminator_graph(
                            fake_BCHW, real_BCHW
                        )
                    if fixed_output is None:
                        fixed_output = self._update_discriminator_fixed_eager(
                            fake_BCHW, real_BCHW
                        )
                    (
                        disc_loss,
                        discriminator_real_metric,
                        discriminator_fake_metric,
                        discriminator_accuracy_metric,
                    ) = fixed_output
                else:
                    logits_fake = self.discriminator_train(
                        self._augment_images(fake_normed_items)
                    )
                    logits_real = self.discriminator_train(
                        self._augment_images(real_normed_items)
                    )
                    discriminator_fake_metric = logits_fake.detach().mean()
                    discriminator_real_metric = logits_real.detach().mean()
                    discriminator_accuracy_metric = (
                        (logits_real.mean(dim=-1) > logits_fake.mean(dim=-1))
                        .float()
                        .mean()
                        .detach()
                    )
                    disc_loss = gan_discriminator_loss(
                        logits_real, logits_fake, gan.discriminator_loss
                    )
                    disc_loss.backward()
                if self._cuda_graphs_enabled:
                    self._sync_discriminator_gradients()
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

    @staticmethod
    def _adaptive_weight(
        reconstruction_loss: torch.Tensor,
        adversarial_loss: torch.Tensor,
        layer: torch.Tensor,
        max_weight: float,
    ) -> torch.Tensor:
        recon_grad = torch.autograd.grad(
            reconstruction_loss, layer, retain_graph=True, allow_unused=True
        )[0]
        gan_grad = torch.autograd.grad(
            adversarial_loss, layer, retain_graph=True, allow_unused=True
        )[0]
        if recon_grad is None or gan_grad is None:
            return reconstruction_loss.new_zeros(())
        return (
            (
                torch.linalg.vector_norm(recon_grad)
                / (torch.linalg.vector_norm(gan_grad) + 1e-6)
            )
            .clamp(0, max_weight)
            .detach()
        )


__all__ = ["RAEStage1Trainer", "RAEGANConfig"]
