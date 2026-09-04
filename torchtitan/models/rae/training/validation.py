# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.validate import BaseValidator, iterate_and_close_dataloader

if TYPE_CHECKING:
    from .trainer import RAEStage1Trainer


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
            if not getattr(collator, "media_kind", None):
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
    def validate(self, model_parts: list[torch.nn.Module], step: int) -> None:
        decoder = model_parts[0]
        was_training = decoder.training
        decoder.eval()
        validation_dataloader = self._validation_dataloader()
        validation_iterator = iter(iterate_and_close_dataloader(validation_dataloader))
        losses: list[torch.Tensor] = []
        comparison_image = None
        num_batches = 0
        num_tokens = 0
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
                self.trainer.metrics_processor.ntokens_since_last_log += batch_tokens
                target_sizes = [
                    tuple(reconstruction.shape[-2:])
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
        }
        if comparison_image is not None:
            extras[
                "validation_images/ground_truth_vs_reconstruction"
            ] = comparison_image
        self.trainer.metrics_processor.log_validation(
            loss=float(torch.stack(losses).mean().item()),
            step=step,
            extra_metrics=extras,
        )


__all__ = ["RAEValidator"]
