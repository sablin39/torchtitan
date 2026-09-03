from types import SimpleNamespace

import pytest
import torch

from torchtitan.components.optimizer.dmuon import load_dmuon
from torchtitan.models.rae.config_registry import rae_stage1_debug
from torchtitan.models.rae.data import RAEQwenCollator
from torchtitan.models.rae.decoder import (
    Cosmos3DRotaryPositionEmbedding,
    create_rae_padding_mask,
    create_rae_varlen_metadata,
    RAEDecoder,
)
from torchtitan.models.rae.discriminator import (
    gan_discriminator_loss,
    gan_generator_loss,
    RAEFeatureDiscriminator,
)
from torchtitan.models.rae.encoder import FrozenRAEEncoder, RAEEncoderConfig
from torchtitan.models.rae.encoder.encoder import _merge_qwen_hidden_states
from torchtitan.models.rae.training import RAEStage1Trainer
from torchtitan.models.rae.training.augmentation import DiscriminatorAugmentation


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


def test_rae_rope_uses_post_merge_centers_and_physical_time() -> None:
    rope = Cosmos3DRotaryPositionEmbedding.Config(
        head_dim=12,
        spatial_merge_size=2,
        temporal_patch_size=2,
        reference_fps=24.0,
    ).build()
    rope.init_states()
    positions = rope.build_positions(
        torch.tensor([2, 2, 3]),
        fps=30.0,
        temporal_start=4.0,
    )
    assert positions.shape == (12, 3)
    torch.testing.assert_close(positions[0], torch.tensor([4.8, 1.0, 1.0]))
    torch.testing.assert_close(positions[1], torch.tensor([4.8, 1.0, 3.0]))
    torch.testing.assert_close(positions[6], torch.tensor([6.4, 1.0, 1.0]))


def test_rae_rope_zero_position_is_identity() -> None:
    rope = Cosmos3DRotaryPositionEmbedding.Config(head_dim=12).build()
    rope.init_states()
    query = torch.randn(5, 3, 12)
    key = torch.randn(5, 3, 12)
    zero_positions = torch.zeros(5, 3)
    rotated_query, rotated_key = rope(query, key, zero_positions)
    torch.testing.assert_close(rotated_query, query)
    torch.testing.assert_close(rotated_key, key)


def test_cosmos_rope_accepts_static_image_fps_zero() -> None:
    rope = Cosmos3DRotaryPositionEmbedding.Config(head_dim=12).build()
    rope.init_states()
    positions = rope.build_positions(
        torch.tensor([1, 2, 2]), fps=0.0, temporal_start=7.0
    )
    assert torch.all(positions[:, 0] == 7.0)
    with pytest.raises(ValueError, match="one-frame"):
        rope.build_positions(torch.tensor([2, 2, 2]), fps=0.0)


def test_rae_decoder_supports_video_grid() -> None:
    model = _debug_decoder()
    latents = torch.randn(1, 2 * 2 * 3, 8)
    output = model(latents, grid_thw=torch.tensor([2, 2, 3]), fps=24.0)
    assert output.shape == (1, 3, 2, 16, 24)


def test_rae_decoder_packed_unpatchify_supports_variable_grids() -> None:
    model = _debug_decoder()
    grid_thw = torch.tensor([[1, 1, 2], [2, 2, 1]])
    latents = torch.randn(int(grid_thw.prod(dim=-1).sum()), 8)
    patch_logits = model(latents, grid_thw=grid_thw)
    outputs = model.unpatchify_packed(
        patch_logits,
        grid_thw,
        patch_size=model.patch_size,
    )
    assert [output.shape for output in outputs] == [(3, 8, 16), (3, 2, 16, 8)]


def test_rae_decoder_packs_unequal_grids_with_equal_token_counts() -> None:
    model = _debug_decoder()
    grid_thw = torch.tensor([[1, 2, 2], [1, 1, 4]])
    latents = torch.randn(int(grid_thw.prod(dim=-1).sum()), 8)
    patch_logits = model(latents, grid_thw=grid_thw)
    outputs = model.unpatchify_packed(
        patch_logits,
        grid_thw,
        patch_size=model.patch_size,
    )
    assert [output.shape for output in outputs] == [(3, 16, 16), (3, 8, 32)]


def test_rae_varlen_metadata_and_padding_mask() -> None:
    metadata = create_rae_varlen_metadata([3, 5])
    assert metadata.cu_seq_q_host == (0, 3, 8)
    assert metadata.cu_seq_q.dtype == torch.int32
    assert metadata.cu_seq_k.dtype == torch.int32
    assert metadata.max_q == metadata.max_k == 5
    mask = create_rae_padding_mask([3, 5])
    assert mask.shape == (2, 5, 5)
    assert mask[0, 0, 2]
    assert not mask[0, 0, 3]


def test_qwen_collator_tracks_post_merge_lengths_and_btchw_media() -> None:
    collator = RAEQwenCollator(
        RAEQwenCollator.Config(batch_size=2, media_kind="video"), context=None
    )
    rows = [
        {
            "pixel_values_videos": torch.randn(8, 6),
            "grid_thw_videos": torch.tensor([[2, 2, 2]]),
            "media": torch.randn(1, 4, 3, 8, 8),
            "merge_size": 2,
            "fps": torch.tensor(12.0),
        },
        {
            "pixel_values_videos": torch.randn(16, 6),
            "grid_thw_videos": torch.tensor([[2, 2, 4]]),
            "media": torch.randn(1, 4, 3, 8, 8),
            "merge_size": 2,
            "fps": torch.tensor(12.0),
        },
    ]
    batch, _ = collator(rows)
    assert batch["input"].shape == (24, 6)
    assert torch.equal(batch["rae_grid_thw"], torch.tensor([[2, 1, 1], [2, 1, 2]]))
    assert torch.equal(batch["sequence_lengths"], torch.tensor([2, 4]))
    assert batch["media"][0].shape == (1, 4, 3, 8, 8)


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


def test_fixed_discriminator_accepts_variable_resolution_images() -> None:
    discriminator = RAEFeatureDiscriminator(
        RAEFeatureDiscriminator.Config(feature_channels=8)
    )
    images = [
        torch.randn(3, 32, 24, requires_grad=True),
        torch.randn(3, 24, 32, requires_grad=True),
    ]
    logits = discriminator(images)
    assert logits.shape == (2, 3)
    logits.mean().backward()
    assert all(image.grad is not None for image in images)


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


def test_variable_supervision_keeps_per_sample_resolution() -> None:
    trainer = object.__new__(RAEStage1Trainer)
    trainer.encoder = SimpleNamespace(supervision_image_size=8)
    images = [torch.rand(3, 16, 24), torch.rand(3, 24, 16)]
    targets = trainer._supervision_images(images, [(4, 6), (6, 4)])
    assert isinstance(targets, list)
    assert [target.shape for target in targets] == [(3, 4, 6), (3, 6, 4)]


def test_debug_recipe_uses_qwen_variable_resolution_and_dmuon() -> None:
    config = rae_stage1_debug()
    assert config.optimizer.param_groups[0].optimizer_name == "DMuon"
    assert config.model_spec.model.use_dmuon
    assert config.model_spec.model.attention_backend == "varlen"
    assert config.encoder.kind == "qwen"
    assert config.dataloader.dataset.processor.image_size is None


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
