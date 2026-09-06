# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


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
    if len(values) != 11:
        raise ValueError(f"RAE Stage 1 metrics require 11 values, got {len(values)}")
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
            "disc_acc=%.5f non_padding=%.5f images_per_step=%.2f%s",
            step,
            *values,
            non_padding_ratio,
            num_images_per_step,
            epoch_suffix,
        )


__all__ = ["log_stage1_metrics"]
