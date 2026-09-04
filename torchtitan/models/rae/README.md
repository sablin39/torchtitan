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

TorchTitan stores the decoder architecture in `config.model_spec.model`, which
is a `RAEDecoder.Config` and is intentionally hidden from the generic CLI.
Stage 1 recipes construct that decoder from the Qwen encoder contract. Both
Qwen and decoder `image_size` fields use `-1` to mean dynamic resolution; the
runtime Qwen `grid_thw` determines the output shape. A positive decoder
`image_size` remains available for legacy fixed-grid calls. The registry names
this optional fixed-grid setting `decoder_image_size`.

The debug recipe uses the checked-in `cc12m_test` images, the local Qwen3.5
vision tower, unequal image grids, packed varlen attention, and DMuon:

```bash
torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
  --module rae --config rae_stage1_debug
```

For graph-enabled throughput, use `rae_stage1_dmuon_static`. Its central
capacity is a 65536-token packed decoder budget, not an image count. Qwen keeps
each image's aspect ratio while constraining its pixel area to at most
1024x1024, and the collator caps each post-merger item at 1024 tokens. It then
packs 63 rows per rank (64512 valid-token capacity), reserving the final 1024
token slots for one isolated FA2 padding document. The decoder never pads a
single image to the full budget:

```bash
torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --module rae --config rae_stage1_dmuon_static
```

The decoder's static tensor capacity is 65536 tokens; the recipe sets
`num_tokens_per_microbatch_per_dp_rank` to 64512 valid tokens and uses 258048
tokens per train step for its four replicas, with one microbatch per rank and a
fixed CUDA-graph input shape. For another replica
count or accumulation setting, use the total-batch equation:

```
global tokens per optimizer step =
    per-rank microbatch tokens * DP degree * accumulation steps
```

For a different DP degree, override
`training.num_tokens_per_train_step` with
`64512 * DP_degree * accumulation_steps`; it must be divisible by
`64512 * DP_degree`. The static row capacity is derived as
`64512 // 1024 = 63` rows per rank. The conservative 65536-token setting leaves
headroom on a 95 GiB RTX PRO 6000; lower the per-rank budget if the available
device has less memory. The HF DINO adversarial input is independently
letterboxed to its
native 224x224 patch grid, so high-resolution decoder outputs do not consume
the discriminator's full-resolution activation memory.

The full OpenImages recipe uses the same Hugging Face streaming source, but
selects every `train_*/*.jpg` folder for training and every `validation*/*.jpg`
folder for validation. It streams image rows without materializing image bytes;
Hugging Face still enumerates matching file paths at startup, so very large
trees may benefit from a prebuilt manifest or shard list. The stream is shuffled
with Grain's bounded window buffer for training, while validation is
deterministic and finite:

```bash
torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
  --module rae --config rae_stage1_openimages
```

The checked-in launcher uses `rae_stage1_openimages_static` for the locally
staged `train_0` and `validation` trees. It keeps the fixed token budget,
compiled decoder/discriminator path, and CUDA graphs:

```bash
torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
  --module rae --config rae_stage1_openimages_static
```

The launcher also enables CUDA allocator expandable segments to reduce
fragmentation when variable-resolution supervision changes the temporary
activation sizes. If your environment already defines
`PYTORCH_CUDA_ALLOC_CONF`, the launcher preserves that value.

After staging all OpenImages folders, use `rae_stage1_openimages` for the
dynamic NAS tree or change that recipe's paths to the complete local tree.

Override `dataloader.streaming_shuffle_buffer_size` to trade startup memory for
shuffle quality. Set `validator.steps` to a positive number for a bounded
validation probe; the default `-1` consumes the validation stream once.

Enable the inherited `validator` section to run RAE reconstruction validation.
The RAE trainer replaces the text validator with an image-aware validator that
uses the configured Qwen media dataloader, reports reconstruction L1 and
post-merger token throughput, and restores the decoder's training mode after
validation. With either `metrics.enable_wandb` or
`metrics.enable_swanlab`, it also logs
`validation_images/ground_truth_vs_reconstruction` as an RGB side-by-side image
(ground truth on the left, reconstruction on the right).

The RAE registry enables both `metrics.enable_wandb` and
`metrics.enable_swanlab` for its recipes. SwanLab receives the metrics through
the WandB-compatible bridge, and the logger forces WandB offline unless
`WANDB_MODE=online` is explicitly set. Install the optional backend with
`pip install swanlab` before launching a registry recipe. For an offline
WandB/SwanLab smoke run:

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --module rae --config rae_stage1_debug --training.steps 2 \
  --validator.enable --validator.steps 2 --validator.freq 1 \
  --metrics.enable-wandb --compile.enable --compile.components model \
  --training.disable-cuda-graphs
```

The varlen recipe uses PyTorch's native `varlen_attn` operator. On SM90/SM100,
TorchTitan selects a registered FA3/FA4 implementation when available. On SM120,
the installed FA4 provider is not compatible, so `VarlenAttention` restores
PyTorch's default FA2 implementation automatically. If a registered FA3/FA4
provider fails to activate, the same FA2 fallback is used. The
`nvidia-cutlass-dsl` package is still needed when the selected provider requires
it; the fallback keeps the recipe functional when that provider is unavailable.

Every packed vision item is an independent attention document. The FA2
`cu_seq_q` and `cu_seq_k` offsets provide that isolation while attention within
each item remains bidirectional (`window_size=(-1, -1)`). No causal mask is
used. SDPA also uses FA2 for unmasked attention when `SDPBackend.FLASH_ATTENTION` is
available. It cannot use FA2 for the dense block-diagonal mask used by the
packed SDPA path: PyTorch rejects non-null masks for its flash kernel and
dispatches to memory-efficient attention instead. Use `attention_backend="varlen"`
for packed unequal-length latents so the FA2 varlen kernel consumes cumulative
sequence offsets without materializing an `T x T` mask.

`--compile.enable --compile.components '["model", "discriminator"]'` compiles
every decoder transformer block and the tensor-only frozen HF DINO forward with
`fullgraph=True`. Variable-grid normalization, position construction,
unpatchification, and image-shape grouping stay in eager wrappers and are
implemented in `layout.py`, `position.py`, and `discriminator/dino.py`. DINO
uses one native 224x224 letterboxed shape, so variable-resolution decoder
outputs share one discriminator graph. The trainable spectral-normalized
discriminator heads remain eager in the non-graph compile path because their
power-iteration buffers are intentionally mutated in place; in CUDA-graph
mode those fixed-shape buffer updates are captured along with head backward.
The frozen backbone is always in evaluation mode.

When CUDA graphs are enabled, `training/graphs.py` captures the fixed-BCHW
reconstruction, perceptual, frozen-discriminator generator loss, and
discriminator forward/backward paths. Graphs are cached by batch and image
shape, and the first occurrence of each shape includes capture and warmup.
Mixed-resolution batches remain on the eager path because a CUDA graph cannot
change tensor shapes. Graph mode bypasses DDP reducer hooks and explicitly
averages discriminator gradients across the batch mesh; this keeps graph
capture safe for replicated DP. DMuon and its optimizer step remain eager,
while the returned reconstruction gradient is propagated through the decoder
normally. On the RTX PRO 6000, a fixed `B=16, 256x256` loss path measured
1.69x faster generator and 3.15x faster discriminator steady-state replay
than eager execution (capture time excluded).

RAE Stage 1 currently uses fully replicated data parallelism. Set
`data_parallel_shard_degree=1`; sharded data parallelism is rejected by the RAE
parallelization entry point because the decoder, packed-token metadata, and
DMuon ownership are not yet implemented for FSDP. Increase
`data_parallel_replicate_degree` with the number of GPUs and set
`training.num_tokens_per_train_step` to the per-rank token budget multiplied by
the replica count and gradient-accumulation steps.

Both Stage 1 recipes load only the vision tower and merger tensors from the
local Qwen3.5-0.8B checkpoint at `~/models/Qwen3.5-0.8B`. Their Qwen processor
leaves dynamic `image_size=-1`, so runtime aspect ratios and resolutions are
retained and the collator packs unequal token grids. The frozen spatial merger
converts the patch grid into post-merger tokens of width 1024. If
`encoder.layer_indices` is non-empty, the selected zero-based block outputs are
summed and passed through the same merger once. A merge size of two gives a
post-merge grid with one quarter of the input spatial token area; the decoder
reconstructs the corresponding runtime resolution.

The decoder uses conservative grouped-query attention (GQA): the base recipe
uses eight query heads and four KV heads with `head_dim=64`, while the debug
recipe uses four query heads and two KV heads. Full Cosmos 3D RoPE is applied to
Q and K independently before the attention kernel groups the KV heads. The
query/KV ratio must remain integral when changing these values. Each block also
applies configurable residual dropout independently after attention and after
the feed-forward branch; the default is `residual_dropout=0.1`, and evaluation
mode disables it.

The default recipe uses the local Hugging Face DINOv3 ViT-L/16 at
`~/models/dinov3-vitl16-pretrain-lvd1689m` as the frozen discriminator backbone,
with intermediate layers 5, 11, 17, and 23 and RAEv2-style residual spectral
heads. The discriminator accepts any compatible local Hugging Face vision model
through `backbone_kind="hf"` and `hf_model_path`; its processor statistics are
read from the model directory. No Python module from the checked-out `RAEv2/`
tree is needed at runtime. DMuon dedicates and replicates its parameter groups
through the regular DDP path.

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

DMuon EMA is maintained as a regular replicated decoder copy. Each generator
update applies the EMA update to that local copy, so there is no FSDP state
gather in the training path.

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
