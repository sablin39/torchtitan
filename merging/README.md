# WSM DCP checkpoint merging

> https://arxiv.org/abs/2507.17634

This directory contains a small offline utility for WSM-style checkpoint
merging. WSM keeps training at the stable post-warmup learning rate and
emulates the effect of a decay phase by averaging recent model checkpoints.
The merged checkpoint is model-only: it is intended for evaluation, export, or
fresh continued training, not for resuming with old optimizer state.

Use source checkpoints from the same model architecture and from a stable-LR
segment. If your run saves DCP checkpoints every 2k steps, a window of 8
checkpoints covers roughly 14k steps of trajectory.

The decay-shaped methods (`linear`, `sqrt`, `cosine`, and `ema`) infer weights
from the actual step numbers in the `step-*` directory names. This means a
partial final checkpoint such as `step-79500` can still be merged sensibly with
regular 2k-step checkpoints. Pass `--expected-interval 2000` so the tool knows
the nominal cadence; interior intervals must match that cadence, while the last
interval may be shorter.

## Dry run

Print the selected checkpoints and merge weights without loading or saving
anything:

```bash
python merging/merge_dcp_checkpoints.py \
  --checkpoint-root outputs/<run>/checkpoint \
  --output-root outputs/<run>/wsm_merges \
  --model-name rwkv_vl \
  --model-flavor 0.4B-v100M \
  --window-sizes 4,8 \
  --methods mean,sqrt \
  --expected-interval 2000 \
  --dry-run
```

## Merge checkpoints

Run the same command without `--dry-run` to write model-only DCP checkpoints:

```bash
python merging/merge_dcp_checkpoints.py \
  --checkpoint-root outputs/<run>/checkpoint \
  --output-root outputs/<run>/wsm_merges \
  --model-name rwkv_vl \
  --model-flavor 0.4B-v100M \
  --window-sizes 4,8 \
  --methods mean,sqrt \
  --expected-interval 2000 \
  --accum-dtype float32 \
  --export-dtype bfloat16
```

Outputs are named by end step, method, and window size, for example:

```text
outputs/<run>/wsm_merges/step-80000_mean_w4/
outputs/<run>/wsm_merges/step-80000_sqrt_w8/
```

Each output contains the DCP checkpoint files plus `merge_metadata.json`, which
records source steps, source paths, weights, method, and dtype settings.

## Convert to Hugging Face

Use the existing converter on any merged DCP checkpoint:

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
  outputs/<run>/wsm_merges/step-80000_sqrt_w8 \
  outputs/<run>/hf_wsm_sqrt_w8 \
  --model_name rwkv_vl \
  --model_flavor 0.4B-v100M \
  --hf_assets_path <hf_assets_path> \
  --export_dtype bfloat16
```

## Recommended first sweep

Start with the paper's strongest practical defaults:

```bash
--methods mean,sqrt
--window-sizes 4,8,12,16
--expected-interval 2000
```

Sweep merge duration first. In the WSM paper, the training span covered by the
merged checkpoints mattered more than the exact number of checkpoints or the
precise averaging formula.

`mean` remains literal uniform checkpoint averaging. Use `linear` when you want
the step-aware version of the theorem-derived weights that matches `mean` only
when the selected checkpoints are evenly spaced.
