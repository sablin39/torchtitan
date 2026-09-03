# RAEv2 Stage 1

This package contains a TorchTitan-native Stage 1 decoder and the model-specific
alternating GAN trainer. The frozen image encoder is kept in the trainer, while
the decoder is built through TorchTitan's meta-device and `Module` protocols.

The package is organized by ownership: `decoder/` contains the transformer,
position encoding, layout, and packing helpers; `encoder/` contains the frozen
Hugging Face vision adapter; `discriminator/` contains the frozen feature
backbones, trainable heads, and perceptual loss; `training/` contains the Stage
1 trainer, DiffAug, and metric logging; `data.py` contains the Qwen media
processor and collator; and `parallelize.py` contains the TorchTitan
parallelization entry point. Qwen processing is the only shipped input path,
so there is no duplicate fixed-image processor or collator.

The debug recipe uses the checked-in `cc12m_test` images, the local Qwen3.5
vision tower, unequal image grids, packed varlen attention, and DMuon:

```bash
torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --module rae --config rae_stage1_debug
```

The varlen recipe uses PyTorch's native `varlen_attn` operator. On SM90/SM100,
TorchTitan selects a registered FA3/FA4 implementation when available. On SM120,
the installed FA4 provider is not compatible, so `VarlenAttention` restores
PyTorch's default FA2 implementation automatically. If a registered FA3/FA4
provider fails to activate, the same FA2 fallback is used. The
`nvidia-cutlass-dsl` package is still needed when the selected provider requires
it; the fallback keeps the recipe functional when that provider is unavailable.

SDPA also uses FA2 for unmasked attention when `SDPBackend.FLASH_ATTENTION` is
available. It cannot use FA2 for the dense block-diagonal mask used by the
packed SDPA path: PyTorch rejects non-null masks for its flash kernel and
dispatches to memory-efficient attention instead. Use `attention_backend="varlen"`
for packed unequal-length latents so the FA2 varlen kernel consumes cumulative
sequence offsets without materializing an `T x T` mask.

`--compile.enable --compile.components '["model"]'` compiles every decoder
transformer block with `fullgraph=True`. Variable-grid normalization,
position construction, and unpatchification stay in the eager wrapper and are
implemented in `layout.py` and `position.py`; this keeps the compiled blocks
tensor-only while still allowing a new graph for each runtime grid shape.

For FSDP training with the DMuon implementation supplied in this checkout,
install the submodule first:

```bash
pip install -e third_party/dmuon
torchrun --standalone --nproc_per_node=8 -m torchtitan.train \
  --module rae --config rae_stage1_dmuon
```

Both Stage 1 recipes load only the vision tower and merger tensors from the
local Qwen3.5-0.8B checkpoint at `~/models/Qwen3.5-0.8B`. Their Qwen processor
leaves `image_size` unset, so runtime aspect ratios and resolutions are retained
and the collator packs unequal token grids. The frozen spatial merger converts
the 16x16 patch grid into 64 post-merger tokens of width 1024. If
`encoder.layer_indices` is non-empty, the selected zero-based block outputs are
summed and passed through the same merger once. A merge size of two gives an
8x8 latent grid and a 128x128 supervision image, so supervision has one quarter
of the encoder input area.

The default recipe uses the local Hugging Face DINOv3 ViT-L/16 at
`~/models/dinov3-vitl16-pretrain-lvd1689m` as the frozen discriminator backbone,
with intermediate layers 5, 11, 17, and 23 and RAEv2-style residual spectral
heads. The discriminator accepts any compatible local Hugging Face vision model
through `backbone_kind="hf"` and `hf_model_path`; its processor statistics are
read from the model directory. No Python module from the checked-out `RAEv2/`
tree is needed at runtime. DMuon dedication runs before `fully_shard`, as
required by its FSDP2 integration.

For RAEv2 parity, set `gan.perceptual_kind="lpips"` and provide
`gan.lpips_calibration_checkpoint_path` (the RAEv2 `vgg.pth` calibration file).
The VGG16 backbone is loaded from torchvision unless
`gan.lpips_vgg_checkpoint_path` points to a local VGG16 state dict. Set
`gan.augment.probability` and `gan.augment.cutout` to the original DiffAug
values; augmentation is applied to both generator and discriminator inputs.
Because DMuon owns backward-time gradient reduction hooks, the optional RAEv2
two-pass adaptive GAN-weight calculation is replaced by a unit multiplier when
DMuon is enabled. This keeps backward and optimizer ordering valid; use AdamW
if exact adaptive-weight parity is required.

DMuon EMA is maintained as a regular, unsharded decoder copy. Each generator
update gathers the full DMuon model state before applying the EMA update, which
is correct but adds communication and memory overhead. A sharded EMA can be
added later if that overhead matters for long runs.

The discriminator begins updating at step 6, two steps before its loss is added
to the generator at step 8. This initializes and warms its spectral-normalized
heads before they supply an adversarial gradient.

The discriminator backbone is permanently frozen. During the generator phase
its trainable heads also have `requires_grad=False`; the discriminator forward
remains differentiable with respect to generated images, so the adversarial
gradient still reaches the decoder. Heads are enabled only for discriminator
updates. The debug recipe uses a lightweight fixed feature pyramid whose heads
mean-pool each runtime feature map, while the full recipe uses the generic
Hugging Face vision discriminator.

## Variable video and resolution

`RAEDecoder` accepts Qwen post-merger latent tokens with explicit
`grid_thw=(T,H,W)` metadata. `H` and `W` are counts after Qwen's spatial
merger; the decoder's `spatial_merge_size` converts their token centers back to
fixed pre-merger patch units. `temporal_patch_size`, `fps`, and
`temporal_start` define a continuous temporal coordinate, so a clip can be
continued without resetting its rotary phase:

```python
reconstruction = decoder(
    latents_BLC,
    grid_thw=torch.tensor([[8, 16, 24]]),
    fps=30.0,
    temporal_start=192.0,
)
```

The decoder uses `Cosmos3DRotaryPositionEmbedding` from `position.py`. Its
frequency allocation, per-axis NTK scaling (default `rope_scale=(2, 1, 1)`),
FPS normalization, and contiguous-half real rotation follow NVIDIA Cosmos'
public implementation as mirrored in Hugging Face Diffusers:
`diffusers/models/transformers/transformer_cosmos.py`. The only RAE adapter is
the Qwen post-merger center coordinate, which preserves the physical footprint
of each merged cell. Images are represented as one-frame media with `fps=0`;
multi-frame inputs require a positive FPS.

For multiple different grids, concatenate the post-merger tokens into `(T,C)`
and construct FA2 metadata from the token counts:

```python
metadata = create_rae_varlen_metadata([8 * 16 * 24, 4 * 12 * 12])
patch_logits = decoder(
    latents_TC,
    grid_thw=torch.tensor([[8, 16, 24], [4, 12, 12]]),
    attention_masks=metadata,
)
clips = decoder.unpatchify_packed(
    patch_logits,
    torch.tensor([[8, 16, 24], [4, 12, 12]]),
    patch_size=decoder.patch_size,
)
```

Set `attention_backend="varlen"` in `RAEDecoder.Config` to route this packed
path through TorchTitan's `VarlenAttention` wrapper (FA2/FA3/FA4 depending on the
active PyTorch kernel). `create_rae_padding_mask` is available for callers that
must retain a padded `(B,L,C)` batch, but packed metadata avoids the padding
overhead and is the recommended path for variable resolution.

TorchTitan's existing `MMSamplePackingConfig` already packs complete
multimodal documents and preserves their image/video lists and position-reset
boundaries. It does not, by itself, batch variable RAE latents: use
`RAEQwenProcessor` and `RAEQwenCollator` (or an equivalent model-side adapter)
to keep Qwen's flattened `pixel_values` and `grid_thw` together. The Stage 1
trainer keeps each image as a list, unpatchifies packed decoder outputs, resizes
each target to its corresponding output grid, and groups only equal shapes for
augmentation. Multi-frame media remains rejected by the 2D GAN trainer until a
video discriminator/loss path is enabled.

`RAEQwenCollator` now also emits `rae_grid_thw` (the post-merger grid),
post-merger `sequence_lengths`, per-item `fps`, and canonical `media` tensors
in `BTCHW` layout. Images use `T=1` and `fps=0`; video rows preserve their frame
rate for the decoder's temporal coordinates. Video FPS must be supplied by the
sample or processor metadata; there is no encoder-level FPS default. The debug
and DMuon recipes use the image path today, while multi-frame media remains
available to packed generation.
