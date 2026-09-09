# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU parity test: torchtitan QwenVisionEncoder vs HF Qwen3_5VisionModel."""

from pathlib import Path

import pytest
import torch

MODEL_DIR = Path("~/models/Qwen3.5-0.8B").expanduser()
LAYER_INDICES = (2, 5, 8, 11)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not MODEL_DIR.is_dir(),
    reason="requires a GPU and the Qwen3.5-0.8B checkpoint",
)


def _load_visual_state() -> dict[str, torch.Tensor]:
    import json

    from safetensors import safe_open

    index_path = MODEL_DIR / "model.safetensors.index.json"
    with index_path.open() as index_file:
        weight_map = json.load(index_file)["weight_map"]
    shard_names = sorted(
        {name for key, name in weight_map.items() if key.startswith("model.visual.")}
    )
    state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        with safe_open(
            str(MODEL_DIR / shard_name), framework="pt", device="cpu"
        ) as shard:
            for key in shard.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual.") :]] = shard.get_tensor(key)
    return state


def _build_hf(state, dtype, attn_implementation):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    full_config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    vision_config = full_config.vision_config
    vision_config._attn_implementation = attn_implementation
    model = Qwen3_5VisionModel(vision_config)
    model.load_state_dict(state, strict=True)
    return model.to(device="cuda", dtype=dtype).eval()


def _build_ours(state, dtype):
    from torchtitan.models.rae.encoder.qwen_vit import (
        QwenVisionConfig,
        QwenVisionEncoder,
    )

    model = QwenVisionEncoder(QwenVisionConfig(), layer_indices=LAYER_INDICES)
    model.load_state_dict(state, strict=True)
    return model.to(device="cuda", dtype=dtype).eval()


def _random_grid_thw(
    num_images: int, generator: torch.Generator, max_side: int = 64
) -> torch.Tensor:
    hw = torch.randint(8, max_side // 2 + 1, (num_images, 2), generator=generator) * 2
    grid_thw = torch.cat([torch.ones(num_images, 1, dtype=torch.long), hw], dim=1)
    return grid_thw


def _pack_pixels(grid_thw: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    num_tokens = int(grid_thw.prod(dim=1).sum().item())
    return torch.randn(num_tokens, 1536, generator=generator)


def _report(name: str, ref: torch.Tensor, out: torch.Tensor) -> float:
    ref = ref.float()
    out = out.float()
    max_abs = (out - ref).abs().max().item()
    mean_abs = (out - ref).abs().mean().item()
    rel = max_abs / ref.abs().max().clamp_min(1e-12).item()
    print(f"{name}: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rel={rel:.6e}")
    return rel


def _report_norm_rel(name: str, ref: torch.Tensor, out: torch.Tensor) -> float:
    ref = ref.float()
    out = out.float()
    norm_rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    print(
        f"{name}: max_abs={(out - ref).abs().max().item():.6e} "
        f"mean_abs={(out - ref).abs().mean().item():.6e} norm_rel={norm_rel:.6e}"
    )
    return norm_rel


def _run_pair(hf_model, tt_model, grid_thw, pixels, dtype):
    pixels = pixels.to(device="cuda", dtype=dtype)
    grid_thw = grid_thw.to(device="cuda")
    with torch.no_grad():
        hf_outputs = hf_model(
            hidden_states=pixels, grid_thw=grid_thw, output_hidden_states=True
        )
        aux = tt_model.build_aux(grid_thw.cpu()).to(torch.device("cuda"))
        final_TD, taps = tt_model(pixels, aux)
        merged = tt_model.merger(final_TD)
    return hf_outputs, final_TD, taps, merged


def _compare_all(hf_outputs, final_TD, taps, merged, rel_tol, tag):
    hf_hidden = hf_outputs.hidden_states
    assert hf_hidden is not None
    worst = 0.0
    for tap, index in zip(taps, LAYER_INDICES):
        worst = max(
            worst,
            _report(f"{tag} block {index}", hf_hidden[index + 1], tap),
        )
    worst = max(worst, _report(f"{tag} final", hf_outputs.last_hidden_state, final_TD))
    worst = max(worst, _report(f"{tag} merger", hf_outputs.pooler_output, merged))
    assert worst < rel_tol, f"{tag} worst rel error {worst} >= {rel_tol}"


@pytest.fixture(scope="module")
def visual_state():
    return _load_visual_state()


def test_bf16_parity(visual_state):
    hf_model = _build_hf(visual_state, torch.bfloat16, "flash_attention_2")
    tt_model = _build_ours(visual_state, torch.bfloat16)
    generator = torch.Generator().manual_seed(0)
    for batch in range(3):
        # max_side 48 keeps peak memory low enough to run alongside training.
        grid_thw = _random_grid_thw(95, generator, max_side=48)
        pixels = _pack_pixels(grid_thw, generator)
        hf_outputs, final_TD, taps, merged = _run_pair(
            hf_model, tt_model, grid_thw, pixels, torch.bfloat16
        )
        _compare_all(hf_outputs, final_TD, taps, merged, 2e-2, f"bf16 batch {batch}")
        del hf_outputs, final_TD, taps, merged
        torch.cuda.empty_cache()


def test_fp32_parity(visual_state):
    # flash-attn is half-precision only; both sides use SDPA in fp32.
    hf_model = _build_hf(visual_state, torch.float32, "sdpa")
    tt_model = _build_ours(visual_state, torch.float32)
    generator = torch.Generator().manual_seed(1)
    grid_thw = _random_grid_thw(8, generator)
    pixels = _pack_pixels(grid_thw, generator)
    hf_outputs, final_TD, taps, merged = _run_pair(
        hf_model, tt_model, grid_thw, pixels, torch.float32
    )
    _compare_all(hf_outputs, final_TD, taps, merged, 1e-4, "fp32")


def test_padding_doc_isolation(visual_state):
    hf_model = _build_hf(visual_state, torch.bfloat16, "flash_attention_2")
    tt_model = _build_ours(visual_state, torch.bfloat16)
    generator = torch.Generator().manual_seed(2)
    grid_thw = _random_grid_thw(16, generator)
    pixels = _pack_pixels(grid_thw, generator)
    num_valid = pixels.shape[0]

    pad_grid = torch.tensor([[1, 16, 16]], dtype=torch.long)
    pad_pixels = torch.randn(256, 1536, generator=generator)
    padded_grid = torch.cat([grid_thw, pad_grid], dim=0)
    padded_pixels = torch.cat([pixels, pad_pixels], dim=0)

    base = _run_pair(hf_model, tt_model, grid_thw, pixels, torch.bfloat16)
    padded = _run_pair(hf_model, tt_model, padded_grid, padded_pixels, torch.bfloat16)
    for name, base_out, padded_out in (
        ("hf final", base[0].last_hidden_state, padded[0].last_hidden_state),
        ("hf merger", base[0].pooler_output, padded[0].pooler_output),
        ("tt final", base[1], padded[1]),
        ("tt merger", base[3], padded[3]),
    ):
        valid_base = base_out[:num_valid]
        valid_padded = padded_out[: len(valid_base)]
        # pooler rows are merged tokens; slice by merged count instead
        if name.endswith("merger"):
            merged_valid = num_valid // 4
            valid_base = base_out[:merged_valid]
            valid_padded = padded_out[:merged_valid]
        rel = _report(f"padding {name}", valid_base, valid_padded)
        assert rel < 1e-3, f"padding doc leaked into valid docs: {name}"


def test_compile_static_shapes(visual_state):
    import torch._inductor.config as inductor_config
    from torch._dynamo.utils import counters

    tt_model = _build_ours(visual_state, torch.bfloat16)
    generator = torch.Generator().manual_seed(3)
    # Same token budget and doc count, different grid splits.
    splits = [
        torch.tensor([[1, 64, 32], [1, 64, 32]], dtype=torch.long),
        torch.tensor([[1, 32, 64], [1, 32, 64]], dtype=torch.long),
    ]
    # emulate_precision_casts keeps inductor fusions at eager rounding; without
    # it compiled-vs-eager bf16 noise on this model's large late-block
    # activation outliers roughly doubles (norm_rel ~8e-2 -> ~3e-2).
    with inductor_config.patch(emulate_precision_casts=True):
        compiled = torch.compile(tt_model, fullgraph=True, dynamic=False)
        counters.clear()
        for split_index, grid_thw in enumerate(splits):
            pixels = _pack_pixels(grid_thw, generator).to(
                device="cuda", dtype=torch.bfloat16
            )
            aux_exact = tt_model.build_aux(grid_thw)
            aux_fixed = tt_model.build_aux(grid_thw, max_seqlen=4096)
            with torch.no_grad():
                eager_final, eager_taps = tt_model(
                    pixels, aux_exact.to(torch.device("cuda"))
                )
                compiled_final, compiled_taps = compiled(
                    pixels, aux_fixed.to(torch.device("cuda"))
                )
            worst = _report_norm_rel(
                f"compiled split {split_index} final", eager_final, compiled_final
            )
            for tap_index, (eager_tap, compiled_tap) in enumerate(
                zip(eager_taps, compiled_taps)
            ):
                worst = max(
                    worst,
                    _report_norm_rel(
                        f"compiled split {split_index} tap {tap_index}",
                        eager_tap,
                        compiled_tap,
                    ),
                )
            # Inductor codegen is not bitwise-identical to eager; this bound
            # matches the model's intrinsic bf16 kernel noise floor (HF flash
            # vs HF sdpa on the same weights differs by norm_rel ~2.7e-2).
            assert worst < 5e-2, f"compiled split {split_index} norm_rel {worst}"
    # torch 2.13 does not populate counters["frames"]; "unique_graphs" is the
    # no-recompile signal: one dynamo graph serves both grid splits.
    unique_graphs = counters["stats"]["unique_graphs"]
    assert unique_graphs == 1, f"expected a single compiled graph, got {unique_graphs}"


def _encoder_config(
    pad_tokens_to: int,
    compile_: bool = False,
    max_tokens_per_doc: int | None = None,
):
    from torchtitan.models.rae.encoder import RAEEncoderConfig

    return RAEEncoderConfig(
        kind="qwen",
        name=str(MODEL_DIR),
        latent_dim=1024,
        image_size=-1,
        layer_indices=LAYER_INDICES,
        merge_size=2,
        dtype="bfloat16",
        compile=compile_,
        pad_tokens_to=pad_tokens_to,
        max_tokens_per_doc=max_tokens_per_doc,
    )


def test_frozen_encoder_integration(visual_state):
    """FrozenRAEEncoder(kind='qwen') with static padding matches the HF path."""
    from torchtitan.models.rae.encoder import FrozenRAEEncoder
    from torchtitan.models.rae.encoder.encoder import _merge_qwen_hidden_states

    generator = torch.Generator().manual_seed(4)
    grid_thw = _random_grid_thw(24, generator, max_side=32)
    pixels = _pack_pixels(grid_thw, generator)
    num_tokens = pixels.shape[0]
    # Static budget: next multiple of 4096, always leaving a nonzero pad doc.
    budget = ((num_tokens + 4095) // 4096) * 4096
    if budget == num_tokens:
        budget += 4096

    encoder = FrozenRAEEncoder(_encoder_config(budget), torch.device("cuda"))
    mapping = {
        "pixel_values": pixels.to(device="cuda", dtype=torch.bfloat16),
        "grid_thw": grid_thw,
    }
    latents, out_grid = encoder(mapping, return_grid_thw=True)

    # HF reference on the real (unpadded) pack, merged with the same helper.
    hf_model = _build_hf(visual_state, torch.bfloat16, "flash_attention_2")
    with torch.no_grad():
        hf_outputs = hf_model(
            hidden_states=mapping["pixel_values"],
            grid_thw=grid_thw.to(device="cuda"),
            output_hidden_states=True,
        )
    taps = [hf_outputs.hidden_states[index + 1] for index in LAYER_INDICES]
    reference = _merge_qwen_hidden_states(
        taps,
        hf_outputs.last_hidden_state,
        hf_model.merger,
        LAYER_INDICES,
        tokens_per_item=grid_thw.prod(dim=-1),
    )

    # The padding document is dropped from both outputs and grid metadata.
    num_real_merged = num_tokens // 4
    assert latents.shape == (num_real_merged, 1024)
    assert out_grid.shape == (grid_thw.shape[0], 3)
    expected_grid = grid_thw.clone()
    expected_grid[:, 1:] //= 2
    assert torch.equal(out_grid.cpu(), expected_grid)
    # When the padding document is the longest varlen segment, flash-attn
    # re-schedules split-kv for the real documents, so the padded pack is not
    # bitwise-identical to the unpadded HF run; the difference is bf16 kernel
    # scheduling noise (compare: HF fa2 vs HF sdpa differ by norm_rel 2.7e-2).
    norm_rel = _report_norm_rel("integration latents", reference, latents)
    assert norm_rel < 2e-2
    assert encoder.last_grid_thw is not None
    assert torch.equal(encoder.last_grid_thw, expected_grid)

    # A tight static max_seqlen pin (covering real docs and the pad doc,
    # which here is below one 4096-token max-size row of slack) only feeds
    # flash-attn kernel scheduling: outputs are bitwise-identical to the
    # budget-wide pin.
    pinned = FrozenRAEEncoder(
        _encoder_config(budget, max_tokens_per_doc=4096), torch.device("cuda")
    )
    pinned_latents, pinned_grid = pinned(mapping, return_grid_thw=True)
    assert torch.equal(pinned_grid, out_grid)
    torch.testing.assert_close(pinned_latents, latents, rtol=0, atol=0)


def test_frozen_encoder_max_tokens_per_doc_too_small(visual_state):
    """A max_seqlen pin below the true longest segment must fail loudly."""
    from torchtitan.models.rae.encoder import FrozenRAEEncoder, RAEEncoderConfig

    generator = torch.Generator().manual_seed(7)
    grid_thw = _random_grid_thw(8, generator, max_side=32)
    pixels = _pack_pixels(grid_thw, generator)
    num_tokens = pixels.shape[0]
    budget = ((num_tokens + 4095) // 4096) * 4096 + 4096
    mapping = {
        "pixel_values": pixels.to(device="cuda", dtype=torch.bfloat16),
        "grid_thw": grid_thw,
    }
    with pytest.raises(ValueError, match="requires pad_tokens_to"):
        RAEEncoderConfig(
            kind="qwen",
            name=str(MODEL_DIR),
            latent_dim=1024,
            image_size=-1,
            merge_size=2,
            max_tokens_per_doc=4096,
        )
    # Real documents reach 32*32=1024 tokens; a pin of 64 is below that.
    encoder = FrozenRAEEncoder(
        _encoder_config(budget, max_tokens_per_doc=64), torch.device("cuda")
    )
    with pytest.raises(ValueError, match="max_seqlen"):
        encoder(mapping)


def test_frozen_encoder_integration_compiled(visual_state):
    """The compiled static-budget encoder matches its eager counterpart."""
    import torch._inductor.config as inductor_config

    from torchtitan.models.rae.encoder import FrozenRAEEncoder

    generator = torch.Generator().manual_seed(5)
    grid_thw = _random_grid_thw(8, generator, max_side=32)
    pixels = _pack_pixels(grid_thw, generator)
    num_tokens = pixels.shape[0]
    budget = ((num_tokens + 4095) // 4096) * 4096
    if budget == num_tokens:
        budget += 4096
    mapping = {
        "pixel_values": pixels.to(device="cuda", dtype=torch.bfloat16),
        "grid_thw": grid_thw,
    }

    eager = FrozenRAEEncoder(_encoder_config(budget), torch.device("cuda"))
    eager_latents, eager_grid = eager(mapping, return_grid_thw=True)
    with inductor_config.patch(emulate_precision_casts=True):
        compiled = FrozenRAEEncoder(
            _encoder_config(budget, compile_=True), torch.device("cuda")
        )
        compiled_latents, compiled_grid = compiled(mapping, return_grid_thw=True)
    assert torch.equal(compiled_grid, eager_grid)
    assert compiled_latents.shape == eager_latents.shape
    norm_rel = _report_norm_rel(
        "integration compiled latents", eager_latents, compiled_latents
    )
    assert norm_rel < 5e-2


def test_frozen_encoder_rejects_over_budget(visual_state):
    from torchtitan.models.rae.encoder import FrozenRAEEncoder

    generator = torch.Generator().manual_seed(6)
    grid_thw = _random_grid_thw(4, generator)
    pixels = _pack_pixels(grid_thw, generator)
    encoder = FrozenRAEEncoder(
        _encoder_config(pixels.shape[0] - 4), torch.device("cuda")
    )
    mapping = {
        "pixel_values": pixels.to(device="cuda", dtype=torch.bfloat16),
        "grid_thw": grid_thw,
    }
    with pytest.raises(ValueError, match="pad_tokens_to"):
        encoder(mapping)


def _build_aux_loop_reference(model, grid_thw, max_seqlen=None, max_docs=None):
    """The original per-document python-loop build_aux, kept as reference."""
    from torchtitan.models.rae.encoder.qwen_vit import _axis_taps_weights, QwenVisionAux

    grid_thw = torch.as_tensor(grid_thw, dtype=torch.long).reshape(-1, 3)
    merge = model.spatial_merge_size
    side = model.num_grid_per_side
    cu_seqlens_list = [0]
    indices_list: list[torch.Tensor] = []
    weights_list: list[torch.Tensor] = []
    positions_list: list[torch.Tensor] = []
    true_max_seqlen = 0
    for t, h, w in grid_thw.tolist():
        seqlen = t * h * w
        true_max_seqlen = max(true_max_seqlen, seqlen)
        for _ in range(t):
            cu_seqlens_list.append(cu_seqlens_list[-1] + h * w)
        within = torch.arange(h * w, dtype=torch.float32)
        blocks_w = w // merge
        in_col = within % merge
        in_row = (within // merge) % merge
        block_col = (within // (merge * merge)) % blocks_w
        block_row = within // (merge * merge * blocks_w)
        row = block_row * merge + in_row
        col = block_col * merge + in_col
        src_h = row * (side - 1) / max(h - 1, 1)
        src_w = col * (side - 1) / max(w - 1, 1)
        h_low, h_wlow, h_high, h_whigh = _axis_taps_weights(src_h, side)
        w_low, w_wlow, w_high, w_whigh = _axis_taps_weights(src_w, side)
        indices_T4 = torch.stack(
            (
                h_low * side + w_low,
                h_low * side + w_high,
                h_high * side + w_low,
                h_high * side + w_high,
            ),
            dim=-1,
        )
        weights_T4 = torch.stack(
            (
                h_wlow * w_wlow,
                h_wlow * w_whigh,
                h_whigh * w_wlow,
                h_whigh * w_whigh,
            ),
            dim=-1,
        )
        positions_T2 = torch.stack((row.long(), col.long()), dim=-1)
        for _ in range(t):
            indices_list.append(indices_T4)
            weights_list.append(weights_T4)
            positions_list.append(positions_T2)
    if max_seqlen is None:
        max_seqlen = true_max_seqlen
    elif max_seqlen < true_max_seqlen:
        raise ValueError(
            f"max_seqlen {max_seqlen} is below the true maximum {true_max_seqlen}"
        )
    if max_docs is not None:
        num_docs = len(cu_seqlens_list) - 1
        if num_docs > max_docs:
            raise ValueError(
                f"batch has {num_docs} documents, above max_docs={max_docs}"
            )
        cu_seqlens_list.extend([cu_seqlens_list[-1]] * (max_docs - num_docs))
    return QwenVisionAux(
        cu_seqlens=torch.tensor(cu_seqlens_list, dtype=torch.int32),
        max_seqlen=max_seqlen,
        interp_indices=torch.cat(indices_list, dim=0),
        interp_weights=torch.cat(weights_list, dim=0),
        position_ids=torch.cat(positions_list, dim=0),
    )


def test_build_aux_vectorized_bit_identical():
    from torchtitan.models.rae.encoder.qwen_vit import (
        QwenVisionConfig,
        QwenVisionEncoder,
    )

    model = QwenVisionEncoder(QwenVisionConfig(), layer_indices=LAYER_INDICES)
    generator = torch.Generator().manual_seed(11)
    for trial in range(5):
        num_docs = int(torch.randint(1, 51, (1,), generator=generator))
        hw = torch.randint(1, 33, (num_docs, 2), generator=generator) * 2
        t = torch.randint(1, 3, (num_docs, 1), generator=generator)
        grid_thw = torch.cat([t, hw], dim=1)
        num_frames = int(t.sum())
        for max_seqlen, max_docs in (
            (None, None),
            (int(grid_thw.prod(dim=-1).max()), None),
            (int(grid_thw.prod(dim=-1).max()) * 3, num_frames + 7),
        ):
            expected = _build_aux_loop_reference(model, grid_thw, max_seqlen, max_docs)
            actual = model.build_aux(grid_thw, max_seqlen, max_docs)
            assert torch.equal(actual.cu_seqlens, expected.cu_seqlens)
            assert actual.max_seqlen == expected.max_seqlen
            assert torch.equal(actual.interp_indices, expected.interp_indices)
            assert torch.equal(actual.interp_weights, expected.interp_weights)
            assert torch.equal(actual.position_ids, expected.position_ids)
    # Error paths preserved.
    with pytest.raises(ValueError, match="spatial_merge_size"):
        model.build_aux(torch.tensor([[1, 3, 4]]))
    with pytest.raises(ValueError, match="max_seqlen"):
        model.build_aux(torch.tensor([[1, 8, 8]]), max_seqlen=63)
    with pytest.raises(ValueError, match="max_docs"):
        model.build_aux(torch.tensor([[1, 2, 2], [1, 2, 2]]), max_docs=1)
