from __future__ import annotations

import copy
from collections.abc import Iterator
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

from .augmentation import DiscriminatorAugmentation
from .discriminator import (
    gan_discriminator_loss,
    gan_generator_loss,
    RAEFeatureDiscriminator,
    RAEPerceptualLoss,
)
from .encoder import FrozenRAEEncoder, RAEEncoderConfig


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
        decoder = self.model_parts[0]
        self._adaptive_weight_enabled = not any(
            hasattr(optimizer, "_dedicated_params") for optimizer in self.optimizers
        )
        self.encoder = FrozenRAEEncoder(config.encoder, self.device)
        if decoder.image_size != self.encoder.supervision_image_size:
            raise ValueError(
                "RAE decoder image_size must equal encoder image_size / merge_size: "
                f"{decoder.image_size} != {self.encoder.supervision_image_size}"
            )
        self.discriminator = RAEFeatureDiscriminator(
            config.discriminator,
            device=self.device,
        ).to(self.device)
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

    def _build_ema(self, decoder: nn.Module):
        try:
            ema_model = copy.deepcopy(decoder).to(self.device).eval()
            ema_model.requires_grad_(False)
            return ema_model, None
        except (RuntimeError, TypeError):
            config = getattr(decoder, "config", None)
            if config is None:
                raise RuntimeError(
                    "RAE EMA could not clone the decoder after parallelization; "
                    "the decoder must expose its config for a materialized EMA copy."
                ) from None
            ema_model = config.build().to(self.device)
            ema_model.init_states()
            if getattr(decoder, "_dmuon_enabled", False) and hasattr(
                decoder, "_dedicated_comm_ctx"
            ):
                dmuon = load_dmuon()
                state = dmuon.get_model_state_dict(
                    decoder, cpu_offload=False, rank0_only=False
                )
                ema_model.load_state_dict(state, strict=False)
            else:
                raise RuntimeError(
                    "RAE EMA cloning failed for a non-DMuon parallelized decoder."
                ) from None
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

    def _next_images(self, data_iterator: Iterator) -> torch.Tensor:
        try:
            input_dict, labels = next(data_iterator)
        except DataloaderExhaustedError:
            raise
        if "input" not in input_dict:
            raise KeyError("RAE Stage 1 batches must contain an 'input' image tensor")
        self.ntokens_seen += labels.numel()
        self.n_valid_tokens_seen += labels.numel()
        self.n_nonpad_tokens_seen += labels.numel()
        return input_dict["input"].to(self.device, non_blocking=True)

    def _supervision_images(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        target_size = self.encoder.supervision_image_size
        if images_BCHW.shape[-2:] == (target_size, target_size):
            return images_BCHW
        return F.interpolate(
            images_BCHW,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)

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
        images_batches: list[torch.Tensor] = []

        self.optimizers.zero_grad(set_to_none=True)
        self.disc_optimizer.zero_grad(set_to_none=True)
        self.discriminator.eval()
        self.discriminator.set_head_requires_grad(False)
        reconstruction_metric = perceptual_metric = adversarial_metric = None
        adaptive_metric = None
        generator_logits_metric = None
        for _ in range(num_microbatches):
            images_BCHW = self._next_images(data_iterator)
            images_batches.append(images_BCHW)
            target_BCHW = self._supervision_images(images_BCHW)
            real_normed_BCHW = target_BCHW * 2.0 - 1.0
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda"
                and self.config.training.dtype == "bfloat16",
            ):
                latents_BCHW = self.encoder(images_BCHW, add_noise=True)
                recon_BCHW = decoder(latents_BCHW)
                recon_normed_BCHW = recon_BCHW * 2.0 - 1.0
                reconstruction_loss = F.l1_loss(recon_BCHW, target_BCHW)
                perceptual_loss = (
                    self.perceptual_loss(real_normed_BCHW, recon_normed_BCHW)
                    if use_perceptual
                    else reconstruction_loss.new_zeros(())
                )
                reconstruction_total = (
                    reconstruction_loss + gan.perceptual_weight * perceptual_loss
                )
                if use_gan:
                    fake_augmented_BCHW = self.discriminator_augmentation(
                        recon_normed_BCHW
                    )
                    logits_fake = self.discriminator_train(fake_augmented_BCHW)
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

        images_BCHW = torch.cat(images_batches, dim=0)
        target_BCHW = self._supervision_images(images_BCHW)
        real_normed_BCHW = target_BCHW * 2.0 - 1.0
        disc_loss = images_BCHW.new_zeros(())
        disc_grad_norm = images_BCHW.new_zeros(())
        discriminator_real_metric = discriminator_fake_metric = images_BCHW.new_zeros(
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
                    fake_BCHW = decoder(self.encoder(images_BCHW)).detach()
                fake_normed_BCHW = (fake_BCHW * 2.0 - 1.0).clamp(-1.0, 1.0)
                fake_normed_BCHW = (
                    torch.round((fake_normed_BCHW + 1.0) * 127.5) / 127.5 - 1.0
                )
                fake_augmented_BCHW = self.discriminator_augmentation(fake_normed_BCHW)
                real_augmented_BCHW = self.discriminator_augmentation(real_normed_BCHW)
                logits_fake = self.discriminator_train(fake_augmented_BCHW)
                logits_real = self.discriminator_train(real_augmented_BCHW)
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
            self._log_stage1_metrics(
                reconstruction_metric,
                perceptual_metric,
                adversarial_metric,
                disc_loss,
                adaptive_metric,
                decoder_grad_norm,
                disc_grad_norm,
                generator_logits_metric
                if generator_logits_metric is not None
                else images_BCHW.new_zeros(()),
                discriminator_real_metric,
                discriminator_fake_metric,
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

    def _log_stage1_metrics(self, *losses: torch.Tensor) -> None:
        values = [float(loss.detach().item()) for loss in losses]
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            from torchtitan.tools.logging import logger

            logger.info(
                "[RAE Stage 1 | step %d] recon=%.5f perceptual=%.5f "
                "gan=%.5f disc=%.5f adaptive=%.5f decoder_grad=%.5f "
                "disc_grad=%.5f gen_logit=%.5f real_logit=%.5f fake_logit=%.5f",
                self.step,
                *values,
            )


__all__ = ["RAEStage1Trainer", "RAEGANConfig"]
