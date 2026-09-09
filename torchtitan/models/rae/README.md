# RAEv2 Stage 1

This package contains a TorchTitan-native Stage 1 decoder and the model-specific
alternating GAN trainer. The frozen image encoder is kept in the trainer, while
the decoder is built through TorchTitan's meta-device and `Module` protocols.

The package is organized by ownership: `decoder.py` contains the transformer,
position encoding, layout, and packing helpers; `encoder.py` contains the frozen
Hugging Face vision adapter; `discriminator/` contains the frozen feature
backbones, trainable heads, and perceptual loss; `trainer.py` contains the Stage
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

For graph-enabled throughput, use `rae_stage1_openimages_static_96k_uvit`. Its
central capacity is a 98304-token packed decoder budget, not an image count.
Qwen keeps each image's aspect ratio while constraining its pixel area to at
most 1024x1024, and the collator caps each post-merger item at 1024 tokens.
Rows are accumulated by token cost until the 97280-token budget would overflow,
so each microbatch carries a variable number of images at ~99% budget fill,
reserving the final 1024 token slots for one isolated FA2 padding document. The
decoder never pads a single image to the full budget:

```bash
torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --module rae --config rae_stage1_openimages_static_96k_uvit
```

The decoder's static tensor capacity is 98304 tokens; the recipe sets
`num_tokens_per_microbatch_per_dp_rank` to 97280 valid tokens and uses 3112960
tokens per train step for its four replicas with eight accumulation
microbatches per rank and a fixed CUDA-graph input shape. For another replica
count or accumulation setting, use the total-batch equation:

```
global tokens per optimizer step =
    per-rank microbatch tokens * DP degree * accumulation steps
```

For a different DP degree, override
`training.num_tokens_per_train_step` with
`97280 * DP_degree * accumulation_steps`; it must be divisible by
`97280 * DP_degree`. Token-budget packing is implemented by a generic
`_TokenBudgetBatchIterDataset` in `torchtitan/components/data/loader.py`,
engaged when a collator exposes `row_cost(row)` and `packing_token_budget()`;
its iterator checkpoints at emitted-batch boundaries so variable row counts
restore exactly. Lower the per-rank budget if the available
device has less memory than a 95 GiB RTX PRO 6000. The HF DINO discriminator
scores images at the decoder's native output resolution, so high-resolution
decoder outputs consume proportionally more discriminator activation memory.

The checked-in launcher uses `rae_stage1_openimages_static_96k_uvit` for the
full OpenImages train set staged as gzipped webdataset tars under
`/mnt/sda1/OpenImages/tar` (16 `train_*.tar.gz` shards, row key `jpg`). The
single-shard `validation.tar.gz` cannot be split across DP ranks, so
validation keeps streaming the locally staged `validation/*.jpg` media folder.
The recipe keeps the fixed token
budget, compiled decoder/discriminator path, and CUDA graphs:

```bash
torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
  --module rae --config rae_stage1_openimages_static_96k_uvit
```

The launcher also enables CUDA allocator expandable segments to reduce
fragmentation when variable-resolution supervision changes the temporary
activation sizes. If your environment already defines
`PYTORCH_CUDA_ALLOC_CONF`, the launcher preserves that value.

The streaming dataloader detects a true epoch: when a rank's DP-sharded
stream exhausts every matched file, its cursor counter wraps and re-shuffles.
`GrainDataLoader.epochs_completed` exposes that counter (per rank). With
`--epochs N` the RAE trainer stops after every rank has completed N epochs
(`training.steps` still bounds the run and sizes the LR/GAN schedule
horizon). Training logs `rae/epoch` each step and, at every epoch boundary,
`rae/tokens_last_epoch`: the valid post-merger tokens the finished epoch
contributed on this rank, which is the measured quantity for refining
tokens-per-epoch schedule estimates. The boundary fires when the last row is
pulled into the pipeline, so up to one shuffle window plus prefetched batches
of the finished epoch may still be in flight and are counted into the next
epoch; both effects are negligible at full-dataset scale.

Override `dataloader.streaming_shuffle_buffer_size` to trade startup memory for
shuffle quality.

The Qwen row processor (JPEG decode, resize, patchify) is CPU-heavy: inline it
sustains only ~25 images/s per rank, well under what the static recipes
consume, so the trainer reports high `time_metrics/data_loading(%)`.
`dataloader.num_processor_workers` (set to 4 in the `_openimages_static`
recipes) fans processing out to spawned worker processes behind a background
pull thread (`ProcessPoolMapIterDataset`), roughly tripling per-rank throughput
while keeping row order, per-row RNG, and checkpoint state identical to the
inline path. It is a suppressed config field -- change it in the recipe, not
via CLI.

Enable the inherited `validator` section to run RAE reconstruction validation.
The RAE trainer replaces the text validator with an image-aware validator that
uses the configured Qwen media dataloader, reports reconstruction L1 and
post-merger token throughput, and restores the decoder's training mode after
validation. Training logs include `rae/non_padding_ratio`, the fraction of the
static decoder token capacity occupied by valid tokens, and
`rae/num_images_per_step`, the number of media items consumed across gradient
accumulation for that optimizer step. Validation reports the corresponding
values under `validation_metrics/`. With either `metrics.enable_wandb` or
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
every decoder transformer block and the frozen HF DINO backbone feature
extraction (`discriminator/dinov3.py`). The discriminator is compiled with
dynamic shapes: images are scored at their native decoder resolution, so
batch, height, and width stay symbolic and variable-resolution microbatches
share one compiled graph without recompilation. The trainable
spectral-normalized heads stay eager by design: their power iteration mutates
the u/v buffers in place on every training-mode forward, and an AOTAutograd
backward captured against those buffers fails its version check once a later
chunk's forward bumps them. Both autograd variants (frozen heads with
grad-enabled generator-side inputs; trainable heads with backward) are warmed
up during trainer initialization, which is covered by
`comm.init_timeout_seconds`, rather than compiling lazily at the first GAN
step. The static recipes loosen the NCCL watchdog to
`init_timeout_seconds=3600` / `train_timeout_seconds=600` because four ranks
compiling concurrently can drift apart by minutes. Variable-grid
normalization, position construction, unpatchification, and image-shape
grouping stay in eager wrappers (`decoder.py`, `discriminator/dinov3.py`). The
frozen backbone is always in evaluation mode.

The decoder keeps its fixed-shape CUDA graphs (`training.disable_cuda_graphs`
controls them). The discriminator does not use CUDA graphs: capture requires
fixed shapes, which conflicts with scoring images at their native
resolutions; the dynamic-shape compiled forward covers the frozen backbone
cost instead. DMuon and its optimizer step remain eager.

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
`encoder.layer_indices` is non-empty, the selected zero-based block outputs
follow RAEv2 multi-layer-sum semantics: each is normalized by the merger's
LayerNorm, the layers are averaged, and the per-item token mean of the final
selected layer is added back as a global signal before one shared merger MLP.
A merge size of two gives a post-merge grid with one quarter of the input
spatial token area; the decoder reconstructs the corresponding runtime
resolution. The tower runs flash-attention varlen over the packed documents in
half precision, and per-document SDPA in fp32.
`encoder.dtype="bfloat16"` casts the frozen weights (the
tower is inference-only). On the RTX PRO 6000 the bf16 + flash combination
measured roughly 7x faster than the original fp32 sdpa path. `encoder.compile`
(torch.compile with fully static shapes under `pad_tokens_to`) is enabled by
the static recipe.

The decoder uses conservative grouped-query attention (GQA): the base recipe
is encoder-sized (~100M parameters: hidden 1024, eight layers, sixteen query
heads and four KV heads with `head_dim=64`, MLP 3072), while the debug recipe
uses four query heads and two KV heads. All linear layers are bias-free and all
normalization is RMSNorm. Full Cosmos 3D RoPE is applied to
Q and K independently before the attention kernel groups the KV heads. The
query/KV ratio must remain integral when changing these values. Each block also
applies configurable residual dropout independently after attention and after
the feed-forward branch; the default is `residual_dropout=0.1`, and evaluation
mode disables it.

The decoder optionally takes U-ViT-style long skip connections through the
`long_skip_connections` config entry, an explicit tuple of `(source, target)`
block pairs: the output of block `source` is concatenated into the input of
block `target` through a dedicated linear projection. The empty default keeps
the plain ViT stack and loads existing checkpoints unchanged. The pairs are
recorded verbatim in the recipe config, so different skip strategies are
comparable from the configuration alone. `rae_stage1_openimages_static_96k_uvit`
runs mirrored pairs `((0, 7), (1, 6), (2, 5), (3, 4))` over the eight blocks.

The default recipe uses the local Hugging Face DINOv3 ViT-B/16 at
`~/models/dinov3-vitb16-pretrain-lvd1689m` as the frozen discriminator backbone,
with intermediate layers 2, 5, 8, and 11 and RAEv2-style residual spectral
heads. ViT-B/16 has a 768-wide hidden state, reducing discriminator feature
memory versus ViT-L/16 while retaining four depth-spaced probes. Images pass
through the backbone unresized at the decoder's output resolution (1/4 of the
encoder input), and each head emits one logit per 16x16 patch token. GAN
losses apply their nonlinearity per patch and reduce image-weighted (per-image
mean first, then batch mean), so large images do not dominate the loss. The
discriminator runs a torchtitan-native DINOv3 ViT-B/16 backbone
(`torchtitan/models/rae/discriminator/dinov3.py`) selected through
`backbone_kind="hf"` and `hf_model_path`; the weights load strictly from the
`model.safetensors` in that directory and inputs get fixed ImageNet
normalization. No Python module from the checked-out `RAEv2/` tree is
needed at runtime. DMuon dedicates and replicates its parameter groups through
the regular DDP path. The recipes apply a decoupled `weight_decay=0.01` to the
Muon-updated matrix parameters (Muon's updates are scale-invariant to the
weight norm, so decay keeps the effective LR from drifting); norm gains and the
cls token route to the AdamW subgroup, which stays decay-free.

For RAEv2 parity, set `gan.perceptual_kind="lpips"` and provide
`gan.lpips_calibration_checkpoint_path` (the RAEv2 `vgg.pth` calibration file).
The VGG16 backbone is loaded from torchvision unless
`gan.lpips_vgg_checkpoint_path` points to a local VGG16 state dict. Set
`gan.augment.probability` and `gan.augment.cutout` to the original DiffAug
values; augmentation is applied to both generator and discriminator inputs.
DMuon owns backward-time gradient reduction hooks, so the RAEv2 two-pass
adaptive GAN-weight calculation wraps its `torch.autograd.grad` probes in
`dmuon.suppress_grad_reduce`, which keeps the probes from dispatching reduces
or consuming the once-per-forward post-backward protocol. The generator loss
"vanilla" is the non-saturating BCE form `softplus(-logit)`; "hinge" is
`-logit`. `gan.discriminator_weight_ramp_steps` ramps the generator-side GAN
weight in linearly after the adversarial phase starts, softening the onset
against an already-converged discriminator. Each discriminator update
processes the step's images in chunks of `gan.discriminator_update_batch_size`
with count-weighted gradients, bounding peak activation memory independently
of how many images a packed step contains.

DMuon EMA is maintained as a regular replicated decoder copy. Each generator
update applies the EMA update to that local copy, so there is no FSDP state
gather in the training path.

The frozen encoder is deterministic, so each microbatch is encoded exactly
once. The generator phase adds RAE noise (one scale per packed media item,
`noise_tau=0.8`) to a copy of the clean latents; the discriminator phase
re-decodes the cached clean latents after the optimizer step, matching RAEv2's
no-grad, noise-free discriminator inputs without a second vision-tower pass.

The GAN phase boundaries follow RAEv2's epoch fractions: discriminator updates
begin at `gan.discriminator_update_start_fraction` (0.375) of training and the
adversarial generator loss at `gan.discriminator_start_fraction` (0.5), so the
first phase trains L1+LPIPS only. `gan.discriminator_*_step` pin absolute step
boundaries instead when set. The discriminator LR warms up linearly over
`gan.discriminator_warmup_steps` and cosine-decays to
`gan.discriminator_final_lr_ratio` of its peak. The early updates initialize
and warm the spectral-normalized heads before they supply an adversarial
gradient.

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

The decoder uses `Cosmos3DRotaryPositionEmbedding` from `decoder.py`. Its
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
active PyTorch kernel); packed metadata avoids the padding overhead of a padded
`(B,L,C)` batch and is the recommended path for variable resolution.

TorchTitan's existing `MMSamplePackingConfig` already packs complete
multimodal documents and preserves their image/video lists and position-reset
boundaries. It does not, by itself, batch variable RAE latents: use
`RAEQwenProcessor` and `RAEQwenCollator` (or an equivalent model-side adapter)
to keep Qwen's flattened `pixel_values` and `grid_thw` together. The Stage 1
trainer keeps each image as a list, unpatchifies packed decoder outputs, resizes
each target to its corresponding output grid, and applies DiffAugment on the
native-resolution discriminator inputs.

`RAEQwenCollator` now also emits `rae_grid_thw` (the post-merger grid),
post-merger `sequence_lengths`, per-item `fps`, and canonical `media` tensors
in `BTCHW` layout. Media is downscaled to at most the resolution the vision
tower saw (grid * patch_size) before collation: supervision targets are the
decoder outputs at half that resolution, so full-resolution originals would
only inflate host memory (previously tens of GiB per prefetched batch).
Images use `T=1` and `fps=0`; multi-frame media remains rejected by the 2D GAN
trainer until a video discriminator/loss path is enabled.
