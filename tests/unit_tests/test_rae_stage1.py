from types import SimpleNamespace

import torch

from torchtitan.components.optimizer.dmuon import load_dmuon
from torchtitan.models.rae.augmentation import DiscriminatorAugmentation
from torchtitan.models.rae.data import RAEImageProcessor
from torchtitan.models.rae.discriminator import (
    gan_discriminator_loss,
    gan_generator_loss,
    RAEFeatureDiscriminator,
)
from torchtitan.models.rae.encoder import (
    _merge_qwen_hidden_states,
    FrozenRAEEncoder,
    RAEEncoderConfig,
)
from torchtitan.models.rae.model import RAEDecoder
from torchtitan.models.rae.trainer import RAEStage1Trainer


def _debug_decoder() -> RAEDecoder:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=32,
        patch_size=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    return model


def test_rae_decoder_forward_shape_and_protocol() -> None:
    model = _debug_decoder()
    model.verify_module_protocol()
    output = model(torch.randn(2, 8, 2, 2))
    assert output.shape == (2, 3, 32, 32)


def test_discriminator_freeze_preserves_fake_image_gradient() -> None:
    discriminator = RAEFeatureDiscriminator(
        RAEFeatureDiscriminator.Config(feature_channels=8)
    )
    discriminator.set_head_requires_grad(False)
    fake = torch.randn(2, 3, 32, 32, requires_grad=True)
    discriminator(fake).mean().backward()
    assert fake.grad is not None
    assert all(parameter.grad is None for parameter in discriminator.heads.parameters())


def test_discriminator_heads_receive_gradients_when_enabled() -> None:
    discriminator = RAEFeatureDiscriminator(
        RAEFeatureDiscriminator.Config(feature_channels=8)
    )
    real = torch.randn(2, 3, 32, 32)
    discriminator(real).mean().backward()
    assert any(
        parameter.grad is not None for parameter in discriminator.heads.parameters()
    )
    assert all(
        parameter.grad is None for parameter in discriminator.backbone.parameters()
    )


def test_frozen_encoder_keeps_input_gradient_path() -> None:
    encoder = FrozenRAEEncoder(
        RAEEncoderConfig(kind="fixed", latent_dim=8, image_size=32),
        torch.device("cpu"),
    )
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    encoder(images).mean().backward()
    assert images.grad is not None
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_qwen_mls_sums_selected_blocks_before_merger() -> None:
    hidden_states = tuple(torch.full((4, 3), float(index)) for index in range(5))
    outputs = SimpleNamespace(
        hidden_states=hidden_states,
        pooler_output=torch.full((1, 3), -1.0),
    )
    merger = torch.nn.Sequential(torch.nn.Identity())
    merged = _merge_qwen_hidden_states(outputs, merger, (0, 2))
    assert torch.equal(merged, hidden_states[1] + hidden_states[3])
    assert torch.equal(
        _merge_qwen_hidden_states(outputs, merger, ()), outputs.pooler_output
    )


def test_merge_size_reduces_supervision_area() -> None:
    trainer = object.__new__(RAEStage1Trainer)
    trainer.encoder = SimpleNamespace(supervision_image_size=8)
    images = torch.rand(2, 3, 16, 16)
    targets = trainer._supervision_images(images)
    assert targets.shape == (2, 3, 8, 8)
    assert (
        targets.shape[-1] * targets.shape[-2]
        == images.shape[-1] * images.shape[-2] // 4
    )


def test_encoder_normalization_statistics_broadcast_channels(tmp_path) -> None:
    stats_path = tmp_path / "stats.pt"
    torch.save(
        {
            "mean": torch.arange(8, dtype=torch.float32),
            "var": torch.ones(8, 2, 2),
        },
        stats_path,
    )
    encoder = FrozenRAEEncoder(
        RAEEncoderConfig(
            kind="fixed",
            latent_dim=8,
            image_size=32,
            normalization_stat_path=str(stats_path),
        ),
        torch.device("cpu"),
    )
    images = torch.rand(1, 3, 32, 32)
    latents = encoder(images)
    assert latents.shape == (1, 8, 2, 2)
    assert torch.isfinite(latents).all()


def test_gan_losses_match_stage1_conventions() -> None:
    logits_real = torch.tensor([[-2.0, 2.0]])
    logits_fake = torch.tensor([[-2.0, 2.0]])
    expected_hinge = 0.5 * (
        torch.relu(1.0 - logits_real).mean() + torch.relu(1.0 + logits_fake).mean()
    )
    assert torch.equal(
        gan_discriminator_loss(logits_real, logits_fake, "hinge"), expected_hinge
    )
    assert torch.equal(gan_generator_loss(logits_fake, "vanilla"), -logits_fake.mean())


def test_vendored_dmuon_is_loadable() -> None:
    dmuon = load_dmuon()
    assert hasattr(dmuon, "Muon")


def test_image_processor_preserves_center_crop_aspect_ratio() -> None:
    processor = object.__new__(RAEImageProcessor)
    processor.image_size = 16
    processor.image_key = "image"
    image = torch.zeros(3, 8, 32)
    image[:, :, 12:20] = 1
    processed = processor(image, None)
    assert processed.shape == (3, 16, 16)
    assert processed.mean() > 0.9


def test_discriminator_augmentation_preserves_generator_gradient() -> None:
    augmentation = DiscriminatorAugmentation(probability=1.0, cutout=0.2)
    images = torch.randn(2, 3, 16, 16, requires_grad=True)
    augmented = augmentation(images)
    augmented.mean().backward()
    assert augmented.shape == images.shape
    assert images.grad is not None


def test_discriminator_augmentation_can_be_disabled() -> None:
    augmentation = DiscriminatorAugmentation(probability=0.0)
    images = torch.randn(2, 3, 16, 16)
    assert torch.equal(augmentation(images), images)
