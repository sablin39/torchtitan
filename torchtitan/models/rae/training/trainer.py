# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator, Mapping
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
from .metrics import log_stage1_metrics
from .validation import RAEValidator


ImageBatch = torch.Tensor | list[torch.Tensor]


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
    discriminator_start_step: int = 8
    discriminator_update_start_step: int = 6
    perceptual_start_step: int = 0
    discriminator_weight: float = 0.75
    perceptual_weight: float = 1.0
    discriminator_updates: int = 1
    generator_loss: str = "vanilla"
    discriminator_loss: str = "hinge"
    max_adaptive_weight: float = 10000.0
    ema_decay: float = 0.9995
    discriminator_lr: float = 2e-4
    discriminator_betas: tuple[float, float] = (0.9, 0.95)
    discriminator_weight_decay: float = 0.0
    discriminator_warmup_steps: int = 0
    perceptual_kind: str = "fixed"
    lpips_calibration_checkpoint_path: str = ""
    lpips_vgg_checkpoint_path: str | None = None
    augment: RAEGANAugmentConfig = field(default_factory=RAEGANAugmentConfig)

    def __post_init__(self) -> None:
        if self.discriminator_updates <= 0:
            raise ValueError("gan.discriminator_updates must be positive")
        if self.discriminator_weight < 0 or self.perceptual_weight < 0:
            raise ValueError("GAN and perceptual weights must be non-negative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("gan.ema_decay must be in [0, 1)")
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
        if (
            self.perceptual_kind == "lpips"
            and not self.lpips_calibration_checkpoint_path
        ):
            raise ValueError(
                "gan.lpips_calibration_checkpoint_path is required for LPIPS"
            )


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

        self.perceptual_loss = RAEPerceptualLoss(
            kind=config.gan.perceptual_kind,
            channels=config.discriminator.feature_channels,
            calibration_checkpoint_path=config.gan.lpips_calibration_checkpoint_path,
            vgg_checkpoint_path=config.gan.lpips_vgg_checkpoint_path,
        ).to(self.device)
        self.discriminator_augmentation = DiscriminatorAugmentation(
            probability=config.gan.augment.probability,
            cutout=config.gan.augment.cutout,
        )
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
        warmup_steps = max(0, config.gan.discriminator_warmup_steps)
        total_steps = max(
            1, config.training.steps * max(1, config.gan.discriminator_updates)
        )

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(
                max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0), 1
            )
            return 1.0 - progress

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

    def _encode_decode(
        self,
        decoder: nn.Module,
        images: ImageBatch,
        encoder_input: Mapping[str, Any] | None,
        *,
        add_noise: bool,
    ) -> list[torch.Tensor]:
        encoder_source = encoder_input if encoder_input is not None else images
        encoded = self.encoder(
            encoder_source,
            add_noise=add_noise,
            return_grid_thw=True,
        )
        if not isinstance(encoded, tuple):
            raise RuntimeError("RAE encoder must return grid metadata for Stage 1")
        latents, grid_thw = encoded
        if add_noise:
            self.metrics_processor.ntokens_since_last_log += int(
                grid_thw.prod(dim=-1).sum().item()
            )
        temporal_start = (
            self.encoder.last_temporal_start
            if self.encoder.last_temporal_start is not None
            else 0.0
        )
        if self._static_sequence_length > 0:
            decoded = self._static_decode(
                decoder,
                latents,
                grid_thw,
                self.encoder.last_fps,
                temporal_start,
            )
        else:
            decoded = decoder(
                latents,
                grid_thw=grid_thw,
                fps=self.encoder.last_fps,
                temporal_start=temporal_start,
            )
        if latents.ndim == 2 or decoded.ndim == 2:
            return decoder.unpatchify_packed(
                decoded,
                grid_thw,
                patch_size=decoder.patch_size,
            )
        return self._image_items(decoded)

    def _perceptual_loss(
        self,
        real_items: list[torch.Tensor],
        fake_items: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = [
            self.perceptual_loss(real_image.unsqueeze(0), fake_image.unsqueeze(0))
            for real_image, fake_image in zip(real_items, fake_items, strict=True)
        ]
        return torch.stack(losses).mean()

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

    def train_step(self, data_iterator: Iterator) -> None:
        decoder = self.model_parts[0]
        gan = self.config.gan
        step = self.step - 1
        use_gan = step >= gan.discriminator_start_step and gan.discriminator_weight > 0
        train_discriminator = (
            step >= gan.discriminator_update_start_step and gan.discriminator_weight > 0
        )
        use_perceptual = step >= gan.perceptual_start_step and gan.perceptual_weight > 0
        num_microbatches = self.gradient_accumulation_steps
        images_batches: list[ImageBatch] = []
        encoder_inputs: list[Mapping[str, Any] | None] = []

        self.optimizers.zero_grad(set_to_none=True)
        self.disc_optimizer.zero_grad(set_to_none=True)
        self.discriminator.eval()
        self.discriminator.set_head_requires_grad(False)
        reconstruction_metric = perceptual_metric = adversarial_metric = None
        adaptive_metric = None
        generator_logits_metric = None
        for _ in range(num_microbatches):
            images, encoder_input = self._next_images(data_iterator)
            images_batches.append(images)
            encoder_inputs.append(encoder_input)
            image_items = self._image_items(images)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda"
                and self.config.training.dtype == "bfloat16",
            ):
                recon_items = self._encode_decode(
                    decoder,
                    images,
                    encoder_input,
                    add_noise=True,
                )
                target_sizes = [
                    tuple(reconstruction.shape[-2:]) for reconstruction in recon_items
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
                perceptual_loss = (
                    self._perceptual_loss(
                        [target * 2.0 - 1.0 for target in target_items],
                        [reconstruction * 2.0 - 1.0 for reconstruction in recon_items],
                    )
                    if use_perceptual
                    else reconstruction_loss.new_zeros(())
                )
                reconstruction_total = (
                    reconstruction_loss + gan.perceptual_weight * perceptual_loss
                )
                if use_gan:
                    fake_augmented = self._augment_images(
                        [reconstruction * 2.0 - 1.0 for reconstruction in recon_items]
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
                        + gan.discriminator_weight * adaptive_weight * adversarial_loss
                    )
                else:
                    adversarial_loss = reconstruction_loss.new_zeros(())
                    adaptive_weight = reconstruction_loss.new_zeros(())
                    total_loss = reconstruction_total
            (total_loss / num_microbatches).backward()
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
        if train_discriminator:
            self.discriminator.set_head_requires_grad(True)
            self.discriminator_train.train()
            for _ in range(gan.discriminator_updates):
                self.disc_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda"
                    and self.config.training.dtype == "bfloat16",
                ):
                    fake_items = [
                        fake.detach()
                        for batch, encoder_input in zip(
                            images_batches, encoder_inputs, strict=True
                        )
                        for fake in self._encode_decode(
                            decoder,
                            batch,
                            encoder_input,
                            add_noise=False,
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
                logits_fake = self.discriminator_train(
                    self._augment_images(fake_normed_items)
                )
                logits_real = self.discriminator_train(
                    self._augment_images(real_normed_items)
                )
                discriminator_fake_metric = logits_fake.detach().mean()
                discriminator_real_metric = logits_real.detach().mean()
                disc_loss = gan_discriminator_loss(
                    logits_real, logits_fake, gan.discriminator_loss
                )
                disc_loss.backward()
                disc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.config.training.max_norm
                )
                self.disc_optimizer.step()
                self.disc_scheduler.step()
            self.discriminator.eval()
            self.discriminator.set_head_requires_grad(False)

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
                ),
                metrics_processor=self.metrics_processor,
            )

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
