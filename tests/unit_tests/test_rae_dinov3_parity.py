# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU parity test: torchtitan DINOv3ViTBackbone vs HF DINOv3ViTModel."""

from pathlib import Path

import pytest
import torch

from torchtitan.models.rae.discriminator.dinov3 import DINOv3ViTBackbone

CHECKPOINT_DIR = Path("~/models/dinov3-vitb16-pretrain-lvd1689m").expanduser()
KEY_DEPTHS = (2, 5, 8, 11)
NUM_PREFIX_TOKENS = 5
# (height, width); non-square included, all sides multiples of 16.
SHAPES = ((224, 224), (256, 192), (112, 336))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (CHECKPOINT_DIR / "model.safetensors").is_file(),
    reason="requires a GPU and the DINOv3 ViT-B/16 checkpoint at " f"{CHECKPOINT_DIR}",
)


def _inputs(height: int, width: int) -> torch.Tensor:
    # Seeded per shape so the HF baseline and the torchtitan module see
    # identical inputs.
    generator = torch.Generator(device="cuda").manual_seed(
        hash((height, width)) % (2**31)
    )
    return torch.randn(2, 3, height, width, device="cuda", generator=generator)


def _extract_hf_features(
    dtype: torch.dtype,
) -> dict[tuple[int, int], list[torch.Tensor]]:
    from transformers import AutoModel

    model = (
        AutoModel.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True)
        .to(dtype=dtype)
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    features = {}
    with torch.no_grad():
        for height, width in SHAPES:
            x_B3HW = _inputs(height, width)
            outputs = model(pixel_values=x_B3HW, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            # Reproduce dino.py's feature extraction: the final normed output
            # plus the normed outputs of blocks at key_depths (hidden_states
            # index d + 1, since index 0 is the embedding output).
            activations_BLC = [outputs.last_hidden_state] + [
                model.norm(hidden_states[depth + 1]) for depth in KEY_DEPTHS
            ]
            features[(height, width)] = [
                activation_BLC[:, NUM_PREFIX_TOKENS:].transpose(1, 2)
                for activation_BLC in activations_BLC
            ]
    return features


@pytest.fixture(scope="module")
def hf_features() -> dict[tuple[int, int], list[torch.Tensor]]:
    return _extract_hf_features(torch.float32)


@pytest.fixture(scope="module")
def hf_features_bf16() -> dict[tuple[int, int], list[torch.Tensor]]:
    return _extract_hf_features(torch.bfloat16)


@pytest.fixture(scope="module")
def titan_backbone() -> DINOv3ViTBackbone:
    return _load_titan_backbone(torch.float32)


@pytest.fixture(scope="module")
def titan_backbone_bf16() -> DINOv3ViTBackbone:
    return _load_titan_backbone(torch.bfloat16)


def _load_titan_backbone(dtype: torch.dtype) -> DINOv3ViTBackbone:
    from safetensors.torch import load_file

    backbone = DINOv3ViTBackbone()
    # Load fp32 weights strictly, then cast to the compute dtype (mirrors
    # dino.py's construction order).
    state = load_file(str(CHECKPOINT_DIR / "model.safetensors"))
    backbone.load_state_dict(state, strict=True)
    backbone.cuda().eval().requires_grad_(False)
    return backbone.to(dtype=dtype)


def _assert_close(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    label: str,
) -> None:
    assert len(actual) == len(expected)
    for index, (actual_BCL, expected_BCL) in enumerate(zip(actual, expected)):
        assert actual_BCL.shape == expected_BCL.shape
        max_diff = (actual_BCL - expected_BCL).abs().max().item()
        scale = expected_BCL.abs().max().item()
        print(
            f"{label} feature {index}: max abs diff {max_diff:.3e}, "
            f"feature abs max {scale:.3e}"
        )
        assert max_diff < 1e-4 * max(scale, 1.0)


def test_parity_with_hf(
    hf_features: dict[tuple[int, int], list[torch.Tensor]],
    titan_backbone: DINOv3ViTBackbone,
) -> None:
    with torch.no_grad():
        for (height, width), expected in hf_features.items():
            actual = titan_backbone(_inputs(height, width), key_depths=KEY_DEPTHS)
            _assert_close(actual, expected, label=f"eager {(height, width)}")


def test_compile_matches_eager(
    hf_features: dict[tuple[int, int], list[torch.Tensor]],
    titan_backbone: DINOv3ViTBackbone,
) -> None:
    compiled = torch.compile(titan_backbone, fullgraph=True, dynamic=True)
    with torch.no_grad():
        for height, width in SHAPES:
            actual = compiled(_inputs(height, width), key_depths=KEY_DEPTHS)
            _assert_close(
                actual,
                hf_features[(height, width)],
                label=f"compiled {(height, width)}",
            )


def test_integrated_discriminator_matches_hf(
    hf_features: dict[tuple[int, int], list[torch.Tensor]],
) -> None:
    from torchtitan.models.rae.discriminator.dinov3 import RAEFeatureDiscriminator

    config = RAEFeatureDiscriminator.Config(
        backbone_kind="hf",
        hf_model_path=str(CHECKPOINT_DIR),
    )
    discriminator = RAEFeatureDiscriminator(config, device=torch.device("cuda")).cuda()
    assert isinstance(discriminator.backbone, DINOv3ViTBackbone)
    with torch.no_grad():
        for height, width in SHAPES[:2]:
            actual = discriminator._backbone_features(_inputs(height, width))
            _assert_close(
                actual,
                hf_features[(height, width)],
                label=f"integrated {(height, width)}",
            )
        # Full forward path: shape grouping, ImageNet normalization, heads.
        logits_BHL = discriminator(torch.rand(2, 3, 224, 224, device="cuda"))
        assert logits_BHL.shape == (2, len(KEY_DEPTHS) + 1, 14 * 14)
        assert torch.isfinite(logits_BHL).all()
        # Eager features() logging path used by feature_distance.
        features = discriminator.features([torch.rand(3, 224, 224, device="cuda")])
        assert len(features) == 1 and len(features[0]) == len(KEY_DEPTHS) + 1


def test_compiled_backward_into_input(
    titan_backbone: DINOv3ViTBackbone,
) -> None:
    # The discriminator backward propagates through the frozen backbone's
    # activations into the (grad-requiring) input image. donated_buffer is
    # disabled because an AOT backward compiled with donated buffers rejects
    # the retain_graph=True double backward used by adaptive-weight probes.
    torch._functorch.config.donated_buffer = False
    compiled = torch.compile(titan_backbone, fullgraph=True, dynamic=True)
    for height, width in SHAPES[:2]:
        x_B3HW = _inputs(height, width).requires_grad_(True)
        features = compiled(x_B3HW, key_depths=KEY_DEPTHS)
        loss = sum(feature_BCL.sum() for feature_BCL in features)
        grad_first = torch.autograd.grad(loss, x_B3HW, retain_graph=True)[0]
        grad_second = torch.autograd.grad(loss, x_B3HW)[0]
        assert torch.equal(grad_first, grad_second)
        assert grad_first.abs().sum() > 0


def test_compile_does_not_respecialize_per_shape(
    titan_backbone: DINOv3ViTBackbone,
) -> None:
    # Regression test: an inductor convolution backward installs
    # shape-equality guards on the frame, recompiling the backbone for every
    # new input resolution until dynamo's recompile limit kills training.
    # The pixel-unshuffle + Linear patch embedding must keep the frame
    # dynamic (one forward graph plus one backward graph, modulo automatic
    # dynamic's static-to-dynamic recompile on the second shape).
    torch._functorch.config.donated_buffer = False
    torch._dynamo.reset()
    compiled = torch.compile(titan_backbone, fullgraph=True, dynamic=True)
    shapes = (
        (224, 224),
        (256, 192),
        (112, 336),
        (128, 128),
        (160, 240),
        (96, 304),
        (320, 176),
        (208, 272),
    )
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    for height, width in shapes:
        x_B3HW = _inputs(height, width).requires_grad_(True)
        features = compiled(x_B3HW, key_depths=KEY_DEPTHS)
        loss = sum(feature_BCL.sum() for feature_BCL in features)
        loss.backward()
    graphs_after = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    num_new_graphs = graphs_after - graphs_before
    print(f"unique dynamo graphs for {len(shapes)} shapes: {num_new_graphs}")
    assert num_new_graphs <= 3


def test_bf16_parity_with_hf(
    hf_features_bf16: dict[tuple[int, int], list[torch.Tensor]],
    titan_backbone_bf16: DINOv3ViTBackbone,
) -> None:
    # Both sides share the bf16 quantization, so the comparison isolates
    # implementation differences rather than rounding. bf16 accumulations
    # (attention, LayerNorm) differ in reduction order, hence a loose
    # relative-norm bound instead of the fp32 elementwise one.
    with torch.no_grad():
        for (height, width), expected in hf_features_bf16.items():
            actual = titan_backbone_bf16(_inputs(height, width), key_depths=KEY_DEPTHS)
            assert len(actual) == len(expected)
            for index, (actual_BCL, expected_BCL) in enumerate(zip(actual, expected)):
                assert actual_BCL.dtype == torch.bfloat16
                diff = actual_BCL.float() - expected_BCL.float()
                norm_rel = (diff.norm() / expected_BCL.float().norm()).item()
                print(
                    f"bf16 {(height, width)} feature {index}: "
                    f"max abs diff {diff.abs().max().item():.3e}, "
                    f"norm rel {norm_rel:.3e}"
                )
                assert norm_rel < 5e-2


def test_bf16_compiled_backward_into_input(
    titan_backbone_bf16: DINOv3ViTBackbone,
) -> None:
    # bf16 variant of the grad-enabled compiled path; see
    # test_compiled_backward_into_input for the donated_buffer rationale.
    torch._functorch.config.donated_buffer = False
    compiled = torch.compile(titan_backbone_bf16, fullgraph=True, dynamic=True)
    for height, width in SHAPES[:2]:
        x_B3HW = _inputs(height, width).requires_grad_(True)
        features = compiled(x_B3HW, key_depths=KEY_DEPTHS)
        loss = sum(feature_BCL.sum() for feature_BCL in features)
        grad_first = torch.autograd.grad(loss, x_B3HW, retain_graph=True)[0]
        grad_second = torch.autograd.grad(loss, x_B3HW)[0]
        assert torch.equal(grad_first, grad_second)
        assert grad_first.abs().sum() > 0


def test_integrated_bf16_discriminator() -> None:
    from torchtitan.models.rae.discriminator.dinov3 import RAEFeatureDiscriminator

    config = RAEFeatureDiscriminator.Config(
        backbone_kind="hf",
        hf_model_path=str(CHECKPOINT_DIR),
        backbone_dtype="bfloat16",
    )
    discriminator = RAEFeatureDiscriminator(config, device=torch.device("cuda")).cuda()
    backbone = discriminator.backbone
    assert backbone.embeddings.patch_embeddings.weight.dtype == torch.bfloat16
    assert next(discriminator.heads.parameters()).dtype == torch.bfloat16
    with torch.no_grad():
        # fp32 input images get normalized in their own dtype and cast to
        # bf16 at the backbone boundary.
        logits_BHL = discriminator(torch.rand(2, 3, 224, 224, device="cuda"))
        assert logits_BHL.dtype == torch.bfloat16
        assert logits_BHL.shape == (2, len(KEY_DEPTHS) + 1, 14 * 14)
        assert torch.isfinite(logits_BHL.float()).all()
        # The logging-only feature distance stays fp32-consistent.
        images = [torch.rand(3, 224, 224, device="cuda") * 2.0 - 1.0]
        distance = discriminator.feature_distance(images, images)
        assert distance.dtype == torch.float32
        assert distance.item() == 0.0
