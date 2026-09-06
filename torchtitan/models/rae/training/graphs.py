# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from ..discriminator import RAEFeatureDiscriminator, RAEPerceptualLoss
from .augmentation import DiscriminatorAugmentation


TensorTuple = tuple[torch.Tensor, ...]


class StaticCUDAGraph:
    """Capture a tensor-only callable and replay it at fixed tensor shapes."""

    def __init__(
        self,
        function: Callable[..., TensorTuple],
    ) -> None:
        self.function = function
        self._graph: torch.cuda.CUDAGraph | None = None
        self._pool: Any = None
        self._static_inputs: tuple[torch.Tensor, ...] | None = None
        self._outputs: TensorTuple | None = None

    @property
    def captured(self) -> bool:
        return self._graph is not None

    def capture(self, *inputs: torch.Tensor) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require CUDA")
        if not inputs or any(not torch.is_tensor(value) for value in inputs):
            raise TypeError("CUDA graph inputs must be tensors")
        device = inputs[0].device
        if device.type != "cuda" or any(value.device != device for value in inputs):
            raise ValueError("CUDA graph inputs must share one CUDA device")
        self._static_inputs = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in inputs
        )

        # Warmup on the capture stream. Backward graphs retain AccumulateGrad
        # stream references, and a side-stream warmup makes those references
        # stale during capture.
        for _ in range(2):
            self.function(*self._static_inputs)
        for value in self._static_inputs:
            if value.grad is not None:
                value.grad.zero_()

        self._pool = torch.cuda.graphs.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # Parameter AccumulateGrad nodes can retain a producer stream from a
        # preceding eager discriminator update. Redirect such stale references
        # to the capture stream instead of rejecting an otherwise valid graph.
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            with torch.cuda.graph(graph, pool=self._pool):
                outputs = self.function(*self._static_inputs)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)
        if not isinstance(outputs, tuple) or not outputs:
            raise TypeError("CUDA graph function must return a non-empty tensor tuple")
        if any(not torch.is_tensor(value) for value in outputs):
            raise TypeError("CUDA graph outputs must be tensors")
        self._graph = graph
        self._outputs = outputs
        self._clear_input_grads()

    def _clear_input_grads(self) -> None:
        if self._static_inputs is None:
            return
        for value in self._static_inputs:
            if value.grad is not None:
                value.grad.zero_()

    def replay(self, *inputs: torch.Tensor) -> TensorTuple:
        if self._graph is None or self._static_inputs is None or self._outputs is None:
            self.capture(*inputs)
        else:
            if len(inputs) != len(self._static_inputs):
                raise ValueError("CUDA graph input count changed between replays")
            with torch.no_grad():
                for static, runtime in zip(self._static_inputs, inputs, strict=True):
                    if (
                        not torch.is_tensor(runtime)
                        or runtime.shape != static.shape
                        or runtime.dtype != static.dtype
                        or runtime.device != static.device
                    ):
                        raise ValueError(
                            "CUDA graph input shape, dtype, or device changed"
                        )
                    static.copy_(runtime)
        self._clear_input_grads()
        assert self._graph is not None
        self._graph.replay()
        assert self._outputs is not None
        return self._outputs


@dataclass(frozen=True, slots=True)
class RAEGeneratorGraphOutput:
    reconstruction_loss: torch.Tensor
    perceptual_loss: torch.Tensor
    adversarial_loss: torch.Tensor
    adaptive_weight: torch.Tensor
    logits_mean: torch.Tensor
    fake_gradient: torch.Tensor


class RAEGeneratorLossGraph:
    """Static loss graph for a homogeneous BCHW generator microbatch.

    The decoder remains outside this graph. The graph differentiates through
    the frozen perceptual network and discriminator, then returns the gradient
    with respect to the reconstruction for eager decoder backpropagation.
    """

    def __init__(
        self,
        discriminator: RAEFeatureDiscriminator,
        perceptual_loss: RAEPerceptualLoss,
        augmentation: DiscriminatorAugmentation,
        *,
        use_gan: bool,
        use_perceptual: bool,
        perceptual_weight: float,
        discriminator_weight: float,
        generator_loss: str,
        loss_scale: float,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        self.discriminator = discriminator
        self.perceptual_loss = perceptual_loss
        self.augmentation = augmentation
        self.use_gan = use_gan
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = discriminator_weight
        self.generator_loss = generator_loss
        self.loss_scale = loss_scale
        self.autocast_dtype = autocast_dtype
        self._graph: StaticCUDAGraph | None = None
        self._shape: tuple[int, ...] | None = None

    def _run(
        self,
        fake_BCHW: torch.Tensor,
        target_BCHW: torch.Tensor,
        valid_image_mask_B: torch.Tensor,
    ) -> TensorTuple:
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )
        with autocast:
            mask_B = valid_image_mask_B.to(dtype=fake_BCHW.dtype)
            normalizer = mask_B.sum().clamp_min(1.0)
            reconstruction_per_image_B = (
                (fake_BCHW - target_BCHW).abs().flatten(1).mean(dim=1)
            )
            reconstruction_loss = (
                reconstruction_per_image_B * mask_B
            ).sum() / normalizer
            if self.use_perceptual:
                perceptual_per_image_B = self.perceptual_loss.forward_per_sample(
                    target_BCHW * 2.0 - 1.0,
                    fake_BCHW * 2.0 - 1.0,
                )
                perceptual_value = (perceptual_per_image_B * mask_B).sum() / normalizer
            else:
                perceptual_value = reconstruction_loss.new_zeros(())
            reconstruction_total = (
                reconstruction_loss + self.perceptual_weight * perceptual_value
            )
            if self.use_gan:
                fake_normed_BCHW = self.augmentation(fake_BCHW * 2.0 - 1.0)
                logits_fake_BH = self.discriminator.forward_fixed(fake_normed_BCHW)
                logits_mean = (logits_fake_BH.mean(dim=1) * mask_B).sum() / normalizer
                if self.generator_loss not in {"hinge", "vanilla"}:
                    raise ValueError(
                        f"Unsupported generator GAN loss: {self.generator_loss}"
                    )
                adversarial_per_image_B = -logits_fake_BH.mean(dim=1)
                adversarial_value = (
                    adversarial_per_image_B * mask_B
                ).sum() / normalizer
            else:
                logits_mean = reconstruction_loss.new_zeros(())
                adversarial_value = reconstruction_loss.new_zeros(())
            adaptive_weight = (
                reconstruction_loss.new_ones(())
                if self.use_gan
                else reconstruction_loss.new_zeros(())
            )
            total_loss = (
                reconstruction_total
                + self.discriminator_weight * adaptive_weight * adversarial_value
            ) * self.loss_scale
        total_loss.backward()
        if fake_BCHW.grad is None:
            raise RuntimeError(
                "Generator CUDA graph did not produce reconstruction gradients"
            )
        return (
            reconstruction_loss.detach(),
            perceptual_value.detach(),
            adversarial_value.detach(),
            adaptive_weight.detach(),
            logits_mean.detach(),
            fake_BCHW.grad,
        )

    def __call__(
        self,
        fake_BCHW: torch.Tensor,
        target_BCHW: torch.Tensor,
        valid_image_mask_B: torch.Tensor,
    ) -> RAEGeneratorGraphOutput:
        if fake_BCHW.ndim != 4 or target_BCHW.shape != fake_BCHW.shape:
            raise ValueError("Generator graph requires matching BCHW tensors")
        if valid_image_mask_B.shape != (fake_BCHW.shape[0],):
            raise ValueError("Generator graph mask must have shape [batch]")
        if not fake_BCHW.requires_grad:
            raise ValueError("Generator graph reconstruction must require gradients")
        shape = tuple(fake_BCHW.shape)
        if self._shape is None:
            self._shape = shape
            self._graph = StaticCUDAGraph(self._run)
        if shape != self._shape:
            raise ValueError("Generator CUDA graph cannot change BCHW shape")
        assert self._graph is not None
        outputs = self._graph.replay(
            fake_BCHW,
            target_BCHW.detach(),
            valid_image_mask_B,
        )
        return RAEGeneratorGraphOutput(*outputs)


@dataclass(frozen=True, slots=True)
class RAEDiscriminatorGraphOutput:
    """Masked sums from one replayed chunk.

    Replays accumulate gradients as unnormalized sums so a variable-size batch
    can be processed in fixed-shape chunks; the caller divides parameters and
    metrics by the accumulated ``valid_count`` once per update.
    """

    loss_sum: torch.Tensor
    real_logits_sum: torch.Tensor
    fake_logits_sum: torch.Tensor
    accuracy_sum: torch.Tensor
    valid_count: torch.Tensor


class RAEDiscriminatorGraph:
    """Static discriminator forward/backward graph for one BCHW shape."""

    def __init__(
        self,
        discriminator: RAEFeatureDiscriminator,
        augmentation: DiscriminatorAugmentation,
        *,
        discriminator_loss: str,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        self.discriminator = discriminator
        self.augmentation = augmentation
        self.discriminator_loss = discriminator_loss
        self.autocast_dtype = autocast_dtype
        self._graph: StaticCUDAGraph | None = None
        self._shape: tuple[int, ...] | None = None

    def _run(
        self,
        fake_BCHW: torch.Tensor,
        real_BCHW: torch.Tensor,
        valid_image_mask_B: torch.Tensor,
    ) -> TensorTuple:
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else nullcontext()
        )
        with autocast:
            fake_normed_BCHW = fake_BCHW.clamp(-1.0, 1.0)
            fake_normed_BCHW = (
                torch.round((fake_normed_BCHW + 1.0) * 127.5) / 127.5 - 1.0
            )
            fake_logits_BH = self.discriminator.forward_fixed(
                self.augmentation(fake_normed_BCHW)
            )
            real_logits_BH = self.discriminator.forward_fixed(
                self.augmentation(real_BCHW)
            )
            mask_B = valid_image_mask_B.to(dtype=fake_logits_BH.dtype)
            if self.discriminator_loss == "hinge":
                loss_per_image_B = 0.5 * (
                    torch.relu(1.0 - real_logits_BH).mean(dim=1)
                    + torch.relu(1.0 + fake_logits_BH).mean(dim=1)
                )
            else:
                loss_per_image_B = 0.5 * (
                    torch.nn.functional.softplus(-real_logits_BH).mean(dim=1)
                    + torch.nn.functional.softplus(fake_logits_BH).mean(dim=1)
                )
            loss_sum = (loss_per_image_B * mask_B).sum()
            real_per_image_B = real_logits_BH.mean(dim=1)
            fake_per_image_B = fake_logits_BH.mean(dim=1)
            accuracy_sum = (
                (real_per_image_B > fake_per_image_B).to(mask_B.dtype) * mask_B
            ).sum()
        loss_sum.backward()
        return (
            loss_sum.detach(),
            (real_per_image_B * mask_B).sum().detach(),
            (fake_per_image_B * mask_B).sum().detach(),
            accuracy_sum.detach(),
            valid_image_mask_B.sum().detach(),
        )

    def __call__(
        self,
        fake_BCHW: torch.Tensor,
        real_BCHW: torch.Tensor,
        valid_image_mask_B: torch.Tensor,
    ) -> RAEDiscriminatorGraphOutput:
        if fake_BCHW.ndim != 4 or real_BCHW.shape != fake_BCHW.shape:
            raise ValueError("Discriminator graph requires matching BCHW tensors")
        if valid_image_mask_B.shape != (fake_BCHW.shape[0],):
            raise ValueError("Discriminator graph mask must have shape [batch]")
        shape = tuple(fake_BCHW.shape)
        if self._shape is None:
            self._shape = shape
            self._graph = StaticCUDAGraph(self._run)
        if shape != self._shape:
            raise ValueError("Discriminator CUDA graph cannot change BCHW shape")
        assert self._graph is not None
        return RAEDiscriminatorGraphOutput(
            *self._graph.replay(
                fake_BCHW.detach(), real_BCHW.detach(), valid_image_mask_B
            )
        )


__all__ = [
    "RAEDiscriminatorGraph",
    "RAEDiscriminatorGraphOutput",
    "RAEGeneratorLossGraph",
    "RAEGeneratorGraphOutput",
    "StaticCUDAGraph",
]
