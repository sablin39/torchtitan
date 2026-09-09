# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch

from torchtitan.components.optimizer.dmuon import load_dmuon
from torchtitan.models.rae.config_registry import (
    model_registry,
    rae_stage1_debug,
    rae_stage1_openimages_static_96k_uvit,
)
from torchtitan.models.rae.data import RAEQwenCollator
from torchtitan.models.rae.decoder import (
    Cosmos3DRotaryPositionEmbedding,
    create_rae_packed_attention_mask,
    create_rae_static_varlen_metadata,
    create_rae_varlen_metadata,
    RAEAttention,
    RAEDecoder,
)
from torchtitan.models.rae.discriminator import (
    gan_discriminator_loss,
    gan_generator_loss,
    RAEFeatureDiscriminator,
    RAEPerceptualLoss,
)
from torchtitan.models.rae.encoder import (
    _merge_qwen_hidden_states,
    FrozenRAEEncoder,
    RAEEncoderConfig,
)
from torchtitan.models.rae.trainer import (
    DiscriminatorAugmentation,
    log_stage1_metrics,
    RAEGANConfig,
    RAEStage1Trainer,
)
from torchtitan.protocols.model_spec import ModelSpec


def _debug_decoder() -> RAEDecoder:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=32,
        patch_size=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    return model


def _debug_varlen_decoder() -> RAEDecoder:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=32,
        patch_size=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        attention_backend="varlen",
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    return model


def _debug_dynamic_decoder() -> RAEDecoder:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=-1,
        patch_size=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
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


def test_rae_decoder_dynamic_size_preserves_rectangular_grid() -> None:
    model = _debug_dynamic_decoder()
    output = model(torch.randn(1, 8, 2, 3))
    assert output.shape == (1, 3, 16, 24)


def test_rae_decoder_uses_gqa_projection_shapes() -> None:
    model = _debug_decoder()
    attention = cast(RAEAttention, model.layers[0].attention)
    assert attention.num_heads == 4
    assert attention.num_kv_heads == 2
    assert attention.enable_gqa
    assert attention.qkv.out_features == 32


def test_rae_decoder_ffn_uses_swiglu() -> None:
    model = _debug_decoder()
    feed_forward = model.layers[0].feed_forward
    inputs = torch.randn(2, 3, 16)
    expected = feed_forward.down(
        torch.nn.functional.silu(feed_forward.gate(inputs)) * feed_forward.up(inputs)
    )
    torch.testing.assert_close(feed_forward(inputs), expected)


def test_rae_decoder_residual_dropout_is_configurable() -> None:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=32,
        patch_size=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        residual_dropout=0.25,
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    assert model.layers[0].residual_dropout == 0.25


def test_rae_ema_builds_unsharded_copy_from_decoder_state() -> None:
    decoder = _debug_decoder()
    trainer = object.__new__(RAEStage1Trainer)
    trainer.device = torch.device("cpu")

    ema_model = trainer._build_ema(decoder)

    assert ema_model is not decoder
    for ema_parameter, parameter in zip(
        ema_model.parameters(), decoder.parameters(), strict=True
    ):
        torch.testing.assert_close(ema_parameter, parameter)
    assert all(not parameter.requires_grad for parameter in ema_model.parameters())


def test_rae_decoder_rejects_invalid_residual_dropout() -> None:
    config = RAEDecoder.Config(residual_dropout=1.0)
    with pytest.raises(ValueError, match="residual_dropout"):
        config.update_from_config(config=type("Config", (), {})())


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


def test_rae_rope_accepts_different_query_and_kv_head_counts() -> None:
    rope = Cosmos3DRotaryPositionEmbedding.Config(head_dim=12).build()
    rope.init_states()
    query = torch.randn(5, 4, 12)
    key = torch.randn(5, 2, 12)
    positions = torch.randn(5, 3)
    rotated_query, rotated_key = rope(query, key, positions)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    torch.testing.assert_close(rotated_query, rope._rotate(query, positions))
    torch.testing.assert_close(rotated_key, rope._rotate(key, positions))


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


def test_rae_decoder_varlen_gqa_forwards_enable_gqa() -> None:
    model = _debug_varlen_decoder()
    assert model.layers[0].attention.varlen_attention.window_size == (-1, -1)
    grid_thw = torch.tensor([[1, 1, 2], [1, 1, 3]])
    latents = torch.randn(5, 8)
    metadata = create_rae_varlen_metadata([2, 3])

    def _identity_varlen(query, key, value, *args, **kwargs):
        assert query.shape == (5, 4, 4)
        assert key.shape == (5, 2, 4)
        assert value.shape == (5, 2, 4)
        assert kwargs["enable_gqa"]
        return query

    with patch(
        "torchtitan.models.common.attention.varlen_attn",
        side_effect=_identity_varlen,
    ):
        patch_logits = model(
            latents,
            grid_thw=grid_thw,
            attention_masks=metadata,
        )
    assert patch_logits.shape == (5, 192)


def test_rae_varlen_metadata() -> None:
    metadata = create_rae_varlen_metadata([3, 5])
    assert metadata.cu_seq_q_host == (0, 3, 8)
    assert metadata.cu_seq_q.dtype == torch.int32
    assert metadata.cu_seq_k.dtype == torch.int32
    assert metadata.max_q == metadata.max_k == 5


def test_rae_packed_attention_mask_is_document_isolated_and_bidirectional() -> None:
    mask = create_rae_packed_attention_mask([2, 3])
    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [False, False, True, True, True],
            [False, False, True, True, True],
            [False, False, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_rae_static_varlen_metadata_reserves_padding_document() -> None:
    metadata = create_rae_static_varlen_metadata([3, 5], 12)
    assert metadata.cu_seq_q.shape == (4,)
    assert metadata.cu_seq_q_host is None
    assert metadata.max_q == metadata.max_k == 12
    assert metadata.cu_seq_q.tolist() == [0, 3, 8, 12]


def test_rae_static_varlen_decoder_fills_multiple_sequences() -> None:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=-1,
        patch_size=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        attention_backend="varlen",
        static_sequence_length=12,
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    grid_thw = torch.tensor([[1, 1, 3], [1, 1, 5]])
    latents = torch.randn(8, 8)

    def _identity_varlen(query, key, value, *args, **kwargs):
        assert query.shape[0] == 12
        assert kwargs["enable_gqa"]
        return query

    with patch(
        "torchtitan.models.common.attention.varlen_attn",
        side_effect=_identity_varlen,
    ):
        padded = model(latents, grid_thw=grid_thw, return_padded=True)
        trimmed = model(latents, grid_thw=grid_thw)
    assert padded.shape == (12, 192)
    assert trimmed.shape == (8, 192)


def test_qwen_collator_tracks_post_merge_lengths_and_btchw_media() -> None:
    collator = RAEQwenCollator(RAEQwenCollator.Config(batch_size=2), context=None)
    rows = [
        {
            "pixel_values": torch.randn(8, 6),
            "grid_thw": torch.tensor([[1, 2, 4]]),
            "media": torch.randn(1, 1, 3, 8, 8),
            "merge_size": 2,
            "fps": torch.tensor(0.0),
        },
        {
            "pixel_values": torch.randn(16, 6),
            "grid_thw": torch.tensor([[1, 4, 4]]),
            "media": torch.randn(1, 1, 3, 8, 8),
            "merge_size": 2,
            "fps": torch.tensor(0.0),
        },
    ]
    batch, _ = collator(rows)
    assert batch["input"].shape == (24, 6)
    assert torch.equal(batch["rae_grid_thw"], torch.tensor([[1, 1, 2], [1, 2, 2]]))
    assert torch.equal(batch["sequence_lengths"], torch.tensor([2, 4]))
    assert batch["media"][0].shape == (1, 1, 3, 8, 8)


def test_qwen_collator_derives_rows_from_token_budget() -> None:
    collator = RAEQwenCollator(
        RAEQwenCollator.Config(
            batch_size=None,
            token_budget=131072,
            max_tokens_per_item=4096,
        ),
        context=None,
    )
    assert collator.num_rows_per_batch() == 32


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
    assert isinstance(logits, list) and len(logits) == 2
    assert all(image_logits.shape[0] == 3 for image_logits in logits)
    torch.stack([image_logits.mean() for image_logits in logits]).sum().backward()
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


def test_qwen_mls_averages_normed_blocks_with_global_mean() -> None:
    torch.manual_seed(0)

    class _Merger(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = 6
            self.use_postshuffle_norm = False
            self.norm = torch.nn.LayerNorm(3)
            self.linear_fc1 = torch.nn.Linear(6, 6)
            self.act_fn = torch.nn.GELU()
            self.linear_fc2 = torch.nn.Linear(6, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm(x).view(-1, self.hidden_size)
            return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

    hidden_states = tuple(torch.randn(4, 3) for _ in range(5))
    merger = _Merger()
    merged = _merge_qwen_hidden_states(
        [hidden_states[1], hidden_states[3]],
        hidden_states[-1],
        merger,
        (0, 2),
        tokens_per_item=torch.tensor([1, 3]),
    )
    normed = [merger.norm(hidden_states[1]), merger.norm(hidden_states[3])]
    averaged = torch.stack(normed).mean(dim=0)
    item_means = torch.stack([normed[1][0], normed[1][1:].mean(dim=0)])
    expected = averaged + torch.repeat_interleave(
        item_means, torch.tensor([1, 3]), dim=0
    )
    expected = merger.linear_fc2(merger.act_fn(merger.linear_fc1(expected.view(-1, 6))))
    torch.testing.assert_close(merged, expected)
    final_hidden = torch.randn(4, 3)
    torch.testing.assert_close(
        _merge_qwen_hidden_states([], final_hidden, merger, ()),
        merger(final_hidden),
    )


def test_qwen_mls_rejects_mismatched_token_counts() -> None:
    hidden_states = tuple(torch.randn(4, 3) for _ in range(5))

    class _Merger(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = 6
            self.use_postshuffle_norm = False
            self.norm = torch.nn.LayerNorm(3)
            self.linear_fc1 = torch.nn.Linear(6, 6)
            self.act_fn = torch.nn.GELU()
            self.linear_fc2 = torch.nn.Linear(6, 3)

    with pytest.raises(ValueError, match="tokens_per_item"):
        _merge_qwen_hidden_states(
            [hidden_states[1], hidden_states[3]],
            hidden_states[-1],
            _Merger(),
            (0, 2),
            tokens_per_item=torch.tensor([2, 2, 2]),
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
    model_spec = cast(ModelSpec, config.model_spec)
    model = cast(RAEDecoder.Config, model_spec.model)
    assert config.optimizer.param_groups[0].optimizer_name == "DMuon"
    assert model.use_dmuon
    assert model.attention_backend == "varlen"
    assert model.num_heads == 4
    assert model.num_kv_heads == 2
    assert config.encoder.kind == "qwen"
    assert config.encoder.image_size == -1
    assert model.image_size == -1


def test_dmuon_recipe_uses_dinov3_vitb16_discriminator() -> None:
    config = rae_stage1_openimages_static_96k_uvit()
    discriminator = config.discriminator
    assert discriminator.backbone_kind == "hf"
    assert discriminator.hf_model_path == "~/models/dinov3-vitb16-pretrain-lvd1689m"
    assert discriminator.feature_channels == 768
    assert discriminator.hf_key_depths == (2, 5, 8, 11)


def test_qwen_collator_row_cost_counts_post_merge_tokens() -> None:
    collator = RAEQwenCollator(
        RAEQwenCollator.Config(
            batch_size=None,
            token_budget=64,
            max_tokens_per_item=16,
        ),
        context=None,
    )
    row = {
        "grid_thw": torch.tensor([[1, 8, 4]]),
        "merge_size": 2,
    }
    assert collator.row_cost(row) == 8
    assert collator.packing_token_budget() == 64
    fixed = RAEQwenCollator(
        RAEQwenCollator.Config(batch_size=2),
        context=None,
    )
    assert fixed.packing_token_budget() is None


def test_token_budget_batching_packs_by_cost_and_restores_state() -> None:
    import grain.python as grain

    from torchtitan.components.data.loader import _TokenBudgetBatchIterDataset

    parent = grain.MapDataset.source(list(range(10))).to_iter_dataset()
    dataset = _TokenBudgetBatchIterDataset(
        parent, batch_fn=list, row_cost=lambda row: 1, token_budget=3
    )
    iterator = iter(dataset)
    assert next(iterator) == [0, 1, 2]
    assert next(iterator) == [3, 4, 5]
    state = iterator.get_state()
    assert next(iterator) == [6, 7, 8]
    replay = iter(dataset)
    next(replay)
    next(replay)
    replay.set_state(state)
    assert next(replay) == [6, 7, 8]
    assert next(iterator) == [9]
    with pytest.raises(StopIteration):
        next(iterator)

    varied = grain.MapDataset.source([2, 2, 2, 1, 1]).to_iter_dataset()
    dataset = _TokenBudgetBatchIterDataset(
        varied, batch_fn=list, row_cost=lambda row: row, token_budget=5
    )
    iterator = iter(dataset)
    assert next(iterator) == [2, 2]
    assert next(iterator) == [2, 1, 1]
    with pytest.raises(StopIteration):
        next(iterator)


def test_discriminator_list_path_groups_shapes_and_matches_loop() -> None:
    torch.manual_seed(0)
    discriminator = RAEFeatureDiscriminator(
        RAEFeatureDiscriminator.Config(feature_channels=8)
    )
    images = [torch.randn(3, 32, 32), torch.randn(3, 24, 40), torch.randn(3, 32, 32)]
    logits_list = discriminator(images)
    assert isinstance(logits_list, list) and len(logits_list) == 3
    # Per-patch logits: one logit per head per feature position.
    assert all(logits.shape[0] == 3 for logits in logits_list)
    assert logits_list[0].shape != logits_list[1].shape
    looped = [discriminator(image.unsqueeze(0))[0] for image in images]
    for got, expected in zip(logits_list, looped, strict=True):
        torch.testing.assert_close(got, expected)
    # Tensor input returns a stacked (B, heads, L) tensor.
    logits_stacked = discriminator(torch.stack([images[0], images[2]]))
    assert isinstance(logits_stacked, torch.Tensor)
    assert logits_stacked.shape[0] == 2
    torch.testing.assert_close(logits_stacked[0], logits_list[0])
    # List and tensor forms of the same batch give identical GAN losses.
    for loss_type in ("hinge", "vanilla"):
        tensor_loss = gan_discriminator_loss(logits_stacked, logits_stacked, loss_type)
        list_loss = gan_discriminator_loss(
            [logits_list[0], logits_list[2]],
            [logits_list[0], logits_list[2]],
            loss_type,
        )
        torch.testing.assert_close(list_loss, tensor_loss)


def test_gan_losses_weight_images_not_patch_tokens() -> None:
    # Constant per-image logits make the expected reduction exact: the
    # image-weighted mean is 1.5, while a token-weighted mean would be 5/3.
    logits_fake = [torch.full((1, 4), 1.0), torch.full((1, 8), 2.0)]
    assert gan_generator_loss(logits_fake, "hinge").item() == pytest.approx(-1.5)
    expected = torch.nn.functional.softplus(torch.tensor([-1.0, -2.0])).mean()
    torch.testing.assert_close(gan_generator_loss(logits_fake, "vanilla"), expected)


def test_chunked_discriminator_update_matches_full_batch() -> None:
    torch.manual_seed(0)
    discriminator = RAEFeatureDiscriminator(
        RAEFeatureDiscriminator.Config(feature_channels=8)
    )
    augmentation = DiscriminatorAugmentation(probability=0.0)
    fake = [torch.randn(3, 16, 16) for _ in range(11)]
    real = [torch.randn(3, 16, 16) for _ in range(11)]

    def run(batch_size: int):
        trainer = object.__new__(RAEStage1Trainer)
        trainer.config = SimpleNamespace(
            gan=RAEGANConfig(discriminator_update_batch_size=batch_size)
        )
        trainer.device = torch.device("cpu")
        trainer.discriminator_train = discriminator
        trainer.discriminator_augmentation = augmentation
        for parameter in discriminator.parameters():
            parameter.grad = None
        output = trainer._update_discriminator(fake, real)
        grads = [
            parameter.grad.clone()
            for parameter in discriminator.parameters()
            if parameter.grad is not None
        ]
        return output, grads

    # A batch size larger than the list degenerates to the full-batch update.
    reference, reference_grads = run(16)
    chunked, chunked_grads = run(4)
    for got, expected in zip(chunked, reference, strict=True):
        torch.testing.assert_close(got, expected)
    for got, expected in zip(chunked_grads, reference_grads, strict=True):
        torch.testing.assert_close(got, expected)


def test_feature_distance_is_zero_for_identical_pairs() -> None:
    discriminator = RAEFeatureDiscriminator(
        RAEFeatureDiscriminator.Config(feature_channels=8)
    )
    real = [torch.randn(3, 32, 32), torch.randn(3, 24, 40)]
    same = discriminator.feature_distance(real, real)
    assert same.item() == pytest.approx(0.0, abs=1e-7)
    different = discriminator.feature_distance(
        real, [torch.randn_like(image) for image in real]
    )
    assert different.item() > 0.0


def test_perceptual_list_path_groups_shapes_and_matches_loop() -> None:
    torch.manual_seed(0)
    loss = RAEPerceptualLoss(kind="fixed", channels=8)
    real_items = [torch.rand(3, 32, 32), torch.rand(3, 24, 40), torch.rand(3, 32, 32)]
    fake_items = [torch.rand_like(item, requires_grad=True) for item in real_items]
    grouped = loss.forward_per_sample_list(real_items, fake_items)
    looped = torch.stack(
        [
            loss.forward_per_sample(real.unsqueeze(0), fake.unsqueeze(0))[0]
            for real, fake in zip(real_items, fake_items)
        ]
    )
    torch.testing.assert_close(grouped, looped)
    grouped.sum().backward()
    assert all(item.grad is not None for item in fake_items)


def test_perceptual_resize_bounds_long_side_and_keeps_grads() -> None:
    torch.manual_seed(0)
    loss = RAEPerceptualLoss(kind="fixed", channels=8, resize_long_side=32)
    real_items = [torch.rand(3, 64, 48), torch.rand(3, 24, 30)]
    fake_items = [torch.rand_like(item, requires_grad=True) for item in real_items]
    grouped = loss.forward_per_sample_list(real_items, fake_items)
    # Small images stay native; both paths must agree with the resize applied.
    looped = torch.stack(
        [
            loss.forward_per_sample(real.unsqueeze(0), fake.unsqueeze(0))[0]
            for real, fake in zip(real_items, fake_items)
        ]
    )
    torch.testing.assert_close(grouped, looped)
    grouped.sum().backward()
    assert all(item.grad is not None for item in fake_items)
    # Zero disables resizing and matches native-resolution evaluation.
    torch.manual_seed(0)
    native = RAEPerceptualLoss(kind="fixed", channels=8, resize_long_side=0)
    reference = native.forward_per_sample_list(real_items, fake_items)
    small_only = RAEPerceptualLoss(kind="fixed", channels=8, resize_long_side=64)
    torch.testing.assert_close(
        small_only.forward_per_sample_list(real_items, fake_items), reference
    )


def test_lpips_list_path_matches_batched_and_skips_real_gradients() -> None:
    vgg_path = "pretrained_models/lpips/vgg16-397923af.pth"
    calibration_path = "pretrained_models/lpips/vgg_lpips.pth"
    if not __import__("pathlib").Path(vgg_path).is_file():
        pytest.skip("LPIPS checkpoints not available")
    torch.manual_seed(0)
    loss = RAEPerceptualLoss(
        kind="lpips",
        calibration_checkpoint_path=calibration_path,
        vgg_checkpoint_path=vgg_path,
    )
    real_items = [torch.rand(3, 32, 32), torch.rand(3, 24, 40), torch.rand(3, 32, 32)]
    fake_items = [torch.rand_like(item) for item in real_items]
    grouped = loss.forward_per_sample_list(real_items, fake_items)
    looped = torch.stack(
        [
            loss.forward_per_sample(real.unsqueeze(0), fake.unsqueeze(0))[0]
            for real, fake in zip(real_items, fake_items)
        ]
    )
    torch.testing.assert_close(grouped, looped, rtol=1e-4, atol=1e-5)
    fake = fake_items[0].clone().requires_grad_(True)
    per_sample = loss.forward_per_sample(real_items[0].unsqueeze(0), fake.unsqueeze(0))
    per_sample.backward()
    assert fake.grad is not None
    assert all(parameter.grad is None for parameter in loss.parameters())


def test_stage1_metrics_include_packing_stats() -> None:
    logged: dict[str, object] = {}

    class Metrics:
        def log(self, *args, **kwargs) -> None:
            del args
            logged.update(kwargs)

    log_stage1_metrics(
        1,
        [torch.zeros(()) for _ in range(12)],
        metrics_processor=Metrics(),
        non_padding_ratio=0.75,
        num_images_per_step=16.0,
    )
    extra_metrics = cast(dict[str, float], logged["extra_metrics"])
    assert extra_metrics["rae/non_padding_ratio"] == 0.75
    assert extra_metrics["rae/num_images_per_step"] == 16.0
    assert "rae/discriminator_accuracy" in extra_metrics


def test_rae_recipe_enables_wandb_and_swanlab_tracking() -> None:
    config = rae_stage1_debug()
    assert config.metrics.enable_wandb
    assert config.metrics.enable_swanlab


def test_static_recipe_packs_multiple_images_into_fixed_token_budget() -> None:
    config = rae_stage1_openimages_static_96k_uvit()
    decoder = cast(RAEDecoder.Config, config.model_spec.model)
    collator = config.dataloader.collator
    validation_collator = config.validator.dataloader.collator
    runtime_collator = RAEQwenCollator(
        collator,
        context=SimpleNamespace(num_tokens_per_batch=97280),
    )
    assert collator.batch_size is None
    assert validation_collator.batch_size is None
    assert runtime_collator.num_rows_per_batch() == 95
    assert collator.token_budget == 97280
    assert collator.max_tokens_per_item == 1024
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 97280
    assert config.training.num_tokens_per_train_step == 3112960
    assert decoder.static_sequence_length == 98304
    assert decoder.attention_backend == "varlen"
    assert not config.training.disable_cuda_graphs
    assert config.compile.components == ["model", "discriminator"]


def test_openimages_static_recipe_keeps_token_budget_and_tar_train_shards() -> None:
    config = rae_stage1_openimages_static_96k_uvit()
    train_source = config.dataloader.dataset.source
    validation_source = config.validator.dataloader.dataset.source
    assert train_source.path == "/mnt/sda1/OpenImages/tar"
    assert train_source.load_dataset_kwargs == {
        "data_files": {"train": "train_*.tar.gz"}
    }
    assert validation_source.path == "/mnt/sda1/OpenImages/validation_subset"
    assert validation_source.load_dataset_kwargs == {
        "data_files": {"train": "validation/*.jpg"}
    }
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 97280
    assert config.training.num_tokens_per_train_step == 3112960
    assert config.validator.enable
    assert config.validator.steps == 16
    assert config.validator.freq == 500


def test_static_recipe_uses_replicated_data_parallelism() -> None:
    config = rae_stage1_openimages_static_96k_uvit()
    assert config.parallelism.data_parallel_replicate_degree == 4
    assert config.parallelism.data_parallel_shard_degree == 1


def test_base_recipe_matches_encoder_sized_gqa_decoder() -> None:
    config = cast(RAEDecoder.Config, model_registry("base").model)
    assert config.latent_dim == 1024
    assert config.image_size == -1
    assert config.patch_size == 16
    assert config.hidden_size == 1024
    assert config.num_layers == 8
    assert config.num_heads == 16
    assert config.num_kv_heads == 4
    assert config.intermediate_size == 3072
    assert config.hidden_size // config.num_heads == 64


def test_base_recipe_decoder_is_bias_free_rmsnorm() -> None:
    model_config = cast(RAEDecoder.Config, model_registry("base").model)
    model_config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = model_config.build()
    linear_modules = [
        module for module in model.modules() if isinstance(module, torch.nn.Linear)
    ]
    assert linear_modules
    assert all(module.bias is None for module in linear_modules)
    assert any(isinstance(module, torch.nn.RMSNorm) for module in model.modules())
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())


def test_model_registry_accepts_explicit_decoder_image_size() -> None:
    config = cast(
        RAEDecoder.Config,
        model_registry("base", decoder_image_size=128).model,
    )
    assert config.image_size == 128


def test_model_registry_accepts_explicit_residual_dropout() -> None:
    config = cast(
        RAEDecoder.Config,
        model_registry("base", residual_dropout=0.2).model,
    )
    assert config.residual_dropout == 0.2


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


def test_gan_fraction_schedule_matches_raev2_phase_boundaries() -> None:
    gan = RAEGANConfig()
    assert gan.discriminator_update_start(10000) == 3750
    assert gan.discriminator_start(10000) == 5000
    pinned = RAEGANConfig(
        discriminator_start_step=8,
        discriminator_update_start_step=6,
    )
    assert pinned.discriminator_update_start(10000) == 6
    assert pinned.discriminator_start(10000) == 8


def test_discriminator_schedule_warms_up_then_cosine_decays() -> None:
    config = rae_stage1_openimages_static_96k_uvit()
    schedule = RAEStage1Trainer._make_discriminator_schedule(config)
    assert schedule(0) == pytest.approx(1.0 / 625)
    assert schedule(624) == pytest.approx(1.0)
    assert schedule(625) == pytest.approx(1.0)
    assert schedule(10000) == pytest.approx(0.1, abs=1e-3)
    midpoint = schedule(5312)
    assert 0.5 < midpoint < 0.6
    # The default config schedules by fraction, so the debug recipe keeps
    # absolute step overrides.
    debug = rae_stage1_debug()
    assert debug.gan.discriminator_start_step == 0
    assert config.gan.discriminator_start_step is None


def test_gan_losses_match_stage1_conventions() -> None:
    logits_real = torch.tensor([[-2.0, 2.0]])
    logits_fake = torch.tensor([[-2.0, 2.0]])
    expected_hinge = 0.5 * (
        torch.relu(1.0 - logits_real).mean() + torch.relu(1.0 + logits_fake).mean()
    )
    assert torch.equal(
        gan_discriminator_loss(logits_real, logits_fake, "hinge"), expected_hinge
    )
    assert torch.equal(gan_generator_loss(logits_fake, "hinge"), -logits_fake.mean())
    assert torch.equal(
        gan_generator_loss(logits_fake, "vanilla"),
        torch.nn.functional.softplus(-logits_fake).mean(),
    )


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


def test_streaming_source_tracks_epoch_completion(tmp_path) -> None:
    from PIL import Image

    from torchtitan.components.data.sources import HuggingFaceStreamingSource
    from torchtitan.components.data.types import DatasetIterationPolicy

    for index in range(4):
        Image.new("RGB", (8, 8), color=(index * 40, 0, 0)).save(
            tmp_path / f"{index}.jpg"
        )
    source = HuggingFaceStreamingSource(
        HuggingFaceStreamingSource.Config(
            path=str(tmp_path),
            split="train",
            load_dataset_kwargs={"data_files": {"train": "*.jpg"}},
        ),
        dataset_iteration_policy=DatasetIterationPolicy(
            seed=42,
            shuffle=False,
            repeat=True,
            dp_rank=0,
            dp_world_size=1,
            streaming_shuffle_buffer_size=4,
        ),
    )
    assert source.current_epoch == 0
    iterator = iter(source)
    for _ in range(4):
        next(iterator)
    # The counter wraps when the first row of the next epoch is pulled.
    assert source.current_epoch == 0
    next(iterator)
    assert source.current_epoch == 1
    state = iterator.get_state()
    for _ in range(4):
        next(iterator)
    assert source.current_epoch == 2
    iterator.set_state(state)
    assert source.current_epoch == 1


def test_streaming_source_can_defer_image_decode(tmp_path) -> None:
    from PIL import Image

    from torchtitan.components.data.sources import HuggingFaceStreamingSource
    from torchtitan.components.data.types import DatasetIterationPolicy
    from torchtitan.models.rae.data import RAEQwenProcessor

    for index in range(2):
        Image.new("RGB", (12, 9), color=(index * 100, 10, 50)).save(
            tmp_path / f"{index}.jpg"
        )
    policy = DatasetIterationPolicy(
        seed=42,
        shuffle=False,
        repeat=False,
        dp_rank=0,
        dp_world_size=1,
        streaming_shuffle_buffer_size=1,
    )
    config_kwargs = {
        "path": str(tmp_path),
        "split": "train",
        "load_dataset_kwargs": {"data_files": {"train": "*.jpg"}},
    }
    decoded = HuggingFaceStreamingSource(
        HuggingFaceStreamingSource.Config(**config_kwargs),
        dataset_iteration_policy=policy,
    )
    deferred = HuggingFaceStreamingSource(
        HuggingFaceStreamingSource.Config(**config_kwargs, decode_images=False),
        dataset_iteration_policy=policy,
    )
    decoded_rows = list(iter(decoded))
    deferred_rows = list(iter(deferred))
    # Filesystem-backed rows carry just the path; tar-backed rows carry the
    # encoded bytes. Both must decode to the reference pixels.
    for row in deferred_rows:
        media = row["image"]
        assert set(media) == {"bytes", "path"}
        assert media["path"] is not None or media["bytes"] is not None
    for decoded_row, deferred_row in zip(decoded_rows, deferred_rows, strict=True):
        reference = RAEQwenProcessor._as_btchw(decoded_row["image"])
        deferred_media = RAEQwenProcessor._as_btchw(deferred_row["image"])
        torch.testing.assert_close(deferred_media, reference)
    # Cursor checkpointing is unaffected by the cast.
    iterator = iter(deferred)
    next(iterator)
    state = iterator.get_state()
    row = next(iterator)
    iterator.set_state(state)
    assert next(iterator)["image"]["path"] == row["image"]["path"]


def test_webdataset_source_defers_image_bytes() -> None:
    from torchtitan.components.data.sources import HuggingFaceStreamingSource
    from torchtitan.components.data.types import DatasetIterationPolicy
    from torchtitan.models.rae.data import RAEQwenProcessor

    source = HuggingFaceStreamingSource(
        HuggingFaceStreamingSource.Config(
            path="tests/assets/cc12m_test",
            split="train",
            load_dataset_kwargs={"data_files": {"train": "cc12m-train-0000.tar"}},
            decode_images=False,
        ),
        dataset_iteration_policy=DatasetIterationPolicy(
            seed=42,
            shuffle=False,
            repeat=False,
            dp_rank=0,
            dp_world_size=1,
            streaming_shuffle_buffer_size=1,
        ),
    )
    row = next(iter(source))
    assert set(row["jpg"]) == {"bytes", "path"}
    assert row["jpg"]["bytes"][:2] == b"\xff\xd8"  # JPEG SOI marker
    media = RAEQwenProcessor._as_btchw(row["jpg"])
    assert media.ndim == 4 and media.shape[1] == 3


def test_find_epoch_source_walks_grain_parents(tmp_path) -> None:
    import grain.python as grain
    from PIL import Image

    from torchtitan.components.data.loader import _find_epoch_source
    from torchtitan.components.data.sources import HuggingFaceStreamingSource
    from torchtitan.components.data.types import DatasetIterationPolicy

    Image.new("RGB", (8, 8)).save(tmp_path / "0.jpg")
    source = HuggingFaceStreamingSource(
        HuggingFaceStreamingSource.Config(
            path=str(tmp_path),
            split="train",
            load_dataset_kwargs={"data_files": {"train": "*.jpg"}},
        ),
        dataset_iteration_policy=DatasetIterationPolicy(
            seed=42,
            shuffle=False,
            repeat=True,
            dp_rank=0,
            dp_world_size=1,
            streaming_shuffle_buffer_size=1,
        ),
    )
    wrapped = source.map(lambda row: row).filter(lambda row: True)
    assert _find_epoch_source(wrapped) is source
    plain = grain.MapDataset.source([1, 2, 3]).to_iter_dataset()
    assert _find_epoch_source(plain) is None


def _epoch_trainer(epochs: int | None, epochs_completed: int | None):
    trainer = object.__new__(RAEStage1Trainer)
    trainer.config = SimpleNamespace(epochs=epochs, training=SimpleNamespace(steps=100))
    trainer.step = 5
    trainer.dataloader = SimpleNamespace(epochs_completed=epochs_completed)
    trainer._tokens_current_epoch = 0
    trainer._last_seen_epoch = 0
    trainer._tokens_last_epoch = None
    trainer._warned_epoch_tracking_missing = False
    return trainer


def test_track_epoch_tokens_records_finished_epoch() -> None:
    trainer = _epoch_trainer(epochs=None, epochs_completed=0)
    trainer._tokens_current_epoch = 500
    assert trainer._track_epoch_tokens() == 0
    assert trainer._tokens_last_epoch is None
    assert trainer._tokens_current_epoch == 500

    trainer._tokens_current_epoch += 700
    trainer.dataloader.epochs_completed = 1
    assert trainer._track_epoch_tokens() == 1
    assert trainer._tokens_last_epoch == 1200
    assert trainer._tokens_current_epoch == 0

    trainer._tokens_current_epoch = 300
    assert trainer._track_epoch_tokens() == 1
    assert trainer._tokens_last_epoch == 1200
    assert trainer._tokens_current_epoch == 300

    # A dataloader without epoch tracking leaves the counters alone.
    untracked = _epoch_trainer(epochs=None, epochs_completed=None)
    untracked._tokens_current_epoch = 42
    assert untracked._track_epoch_tokens() is None
    assert untracked._tokens_last_epoch is None
    assert untracked._tokens_current_epoch == 42


def test_epochs_stop_uses_dataloader_epoch_counter() -> None:
    trainer = _epoch_trainer(epochs=2, epochs_completed=1)
    assert trainer.should_continue_training()
    trainer.dataloader.epochs_completed = 2
    assert not trainer.should_continue_training()

    # The steps cap still applies when epochs are set.
    trainer.dataloader.epochs_completed = 0
    trainer.step = 100
    assert not trainer.should_continue_training()

    # No epochs configured: step-only training.
    trainer = _epoch_trainer(epochs=None, epochs_completed=5)
    trainer.step = 5
    assert trainer.should_continue_training()

    # Untracked dataloader: warn once and keep training.
    trainer = _epoch_trainer(epochs=2, epochs_completed=None)
    assert trainer.should_continue_training()
    assert trainer._warned_epoch_tracking_missing


def _debug_uvit_decoder(
    long_skip_connections: tuple[tuple[int, int], ...],
) -> RAEDecoder:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=32,
        patch_size=8,
        hidden_size=16,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        long_skip_connections=long_skip_connections,
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    return model


def test_rae_decoder_long_skip_connections_forward_shape() -> None:
    model = _debug_uvit_decoder(((0, 3), (1, 2)))
    assert len(model.skip_projections) == 2
    assert "skip_projections.0.weight" in model.state_dict()
    output = model(torch.randn(2, 8, 2, 2))
    assert output.shape == (2, 3, 32, 32)


def test_rae_decoder_long_skip_connections_route_declared_pairs() -> None:
    model = _debug_uvit_decoder(((0, 3), (1, 2)))

    class _IncrementBlock(torch.nn.Module):
        def forward(self, hidden, *, positions, attention_masks):
            del positions, attention_masks
            return hidden + 1

    model.layers = torch.nn.ModuleList([_IncrementBlock() for _ in range(4)])
    # Identity concat projections: proj(cat([a, b])) = a + b.
    with torch.no_grad():
        for projection in model.skip_projections:
            projection.weight.zero_()
            projection.weight[:, :16] = torch.eye(16)
            projection.weight[:, 16:] = torch.eye(16)

    hidden_BLD = torch.randn(2, 5, 16)
    output = model._apply_blocks(
        hidden_BLD, positions=torch.zeros(2, 5, 3), attention_masks=None
    )
    # h1=h+2 enters block 2 paired with skip1=h+2 -> 2h+5 after block 2;
    # then paired with skip0=h+1 -> 3h+7 after block 3.
    torch.testing.assert_close(output, 3 * hidden_BLD + 7)


def test_rae_decoder_rejects_invalid_long_skip_connections() -> None:
    invalid = [
        ((1, 1),),
        ((3, 1),),
        ((0, 4),),
        ((0, 2), (1, 2)),
    ]
    for pairs in invalid:
        config = RAEDecoder.Config(num_layers=4, long_skip_connections=pairs)
        with pytest.raises(ValueError, match="long_skip_connections"):
            config.update_from_config(config=type("Config", (), {})())


def test_rae_decoder_default_has_no_skip_projection_parameters() -> None:
    model = _debug_decoder()
    assert not any("skip_projections" in key for key in model.state_dict())


def test_openimages_static_uvit_recipe_records_skip_pairs_and_weight_decay() -> None:
    config = rae_stage1_openimages_static_96k_uvit()
    decoder = cast(RAEDecoder.Config, config.model_spec.model)
    assert decoder.long_skip_connections == ((0, 7), (1, 6), (2, 5), (3, 4))
    optimizer_kwargs = config.optimizer.param_groups[0].optimizer_kwargs
    assert optimizer_kwargs["weight_decay"] == 0.01
    assert optimizer_kwargs["adamw_weight_decay"] == 0.0
