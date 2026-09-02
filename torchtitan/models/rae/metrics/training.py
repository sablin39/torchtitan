from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


def log_stage1_metrics(step: int, losses: Sequence[torch.Tensor]) -> None:
    """Log the scalar metrics emitted by one RAE Stage 1 update."""
    values = [float(loss.detach().item()) for loss in losses]
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            "[RAE Stage 1 | step %d] recon=%.5f perceptual=%.5f "
            "gan=%.5f disc=%.5f adaptive=%.5f decoder_grad=%.5f "
            "disc_grad=%.5f gen_logit=%.5f real_logit=%.5f fake_logit=%.5f",
            step,
            *values,
        )


__all__ = ["log_stage1_metrics"]
