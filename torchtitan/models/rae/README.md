# RAEv2 Stage 1

This package contains a TorchTitan-native Stage 1 decoder and the model-specific
alternating GAN trainer. The frozen image encoder is kept in the trainer, while
the decoder is built through TorchTitan's meta-device and `Module` protocols.

The debug recipe uses the checked-in `cc12m_test` images and a deterministic
fixed encoder:

```bash
torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --module rae --config rae_stage1_debug
```

For FSDP training with the DMuon implementation supplied in this checkout,
install the submodule first:

```bash
pip install -e third_party/dmuon
torchrun --standalone --nproc_per_node=8 -m torchtitan.train \
  --module rae --config rae_stage1_dmuon
```

`rae_stage1_dmuon` loads only the vision tower and merger tensors from the local
Qwen3.5-0.8B checkpoint at `~/models/Qwen3.5-0.8B`. The Qwen processor receives
the 256x256 input without a second rescale. The frozen spatial merger converts
the 16x16 patch grid into 64 post-merger tokens of width 1024. If
`encoder.layer_indices` is non-empty, the selected zero-based block outputs are
summed and passed through the same merger once. A merge size of two gives an
8x8 latent grid and a 128x128 supervision image, so supervision has one quarter
of the encoder input area.

The default recipe uses the local DINOv3 ViT-L/16 at
`~/models/dinov3-vitl16-pretrain-lvd1689m` as the frozen discriminator backbone,
with intermediate layers 5, 11, 17, and 23 and RAEv2-style residual spectral
heads. Override `encoder.name` and `discriminator.dino_model_path` when either
asset is stored elsewhere. For strict original-critic parity, select
`backbone_kind="dinodisc"` and provide the DINO-S/8 checkpoint through
`dino_ckpt_path`. No Python module from the checked-out `RAEv2/` tree is needed
at runtime. DMuon dedication runs before `fully_shard`, as required by its
FSDP2 integration.

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
updates.
