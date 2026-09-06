
set -euo pipefail


cd /home/rwkv/molin/torchtitan
source .venv/bin/activate



export WANDB_PROJECT=rae-stage1
export WANDB_RUN_NAME=rae-openimages-static-$(date +%Y%m%d-%H%M%S)
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

N_EPOCHS=10
DP_DEGREE=4
ACCUM_STEPS=8
# 96k static recipe: 98304-token padded capacity, 97280-token packed budget.
TOKENS_PER_MICROBATCH=97280
# Training stops after N_EPOCHS full passes over the dataset (--epochs).
# TOKENS_PER_EPOCH is only an estimate that sizes the step-based LR/GAN
# schedule horizon and the steps safety cap; the measured tokens per epoch
# are logged as rae/tokens_last_epoch so the estimate can be refined.
# Estimate: 1.75M train images (per-tar member counts sampled from the first
# 2GB of each tar and scaled by file size; validated 0.3% against the exact
# train_0 folder count) x 729 post-merge tokens/image measured through the
# recipe's Qwen processor on a 400-image sample.
TOKENS_PER_EPOCH=1274000000
RAE_CONFIG=${RAE_CONFIG:-rae_stage1_openimages_static_96k}

TOKENS_PER_STEP=$((TOKENS_PER_MICROBATCH * DP_DEGREE * ACCUM_STEPS))
TRAINING_STEPS=$(((
    N_EPOCHS * TOKENS_PER_EPOCH + TOKENS_PER_STEP - 1
) / TOKENS_PER_STEP))

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone \
  --nproc_per_node="${DP_DEGREE}" \
  -m torchtitan.train \
  --module rae \
  --config "${RAE_CONFIG}" \
  --epochs "${N_EPOCHS}" \
  --training.num_tokens_per_microbatch_per_dp_rank "${TOKENS_PER_MICROBATCH}" \
  --training.num_tokens_per_train_step "${TOKENS_PER_STEP}" \
  --training.steps "${TRAINING_STEPS}" \
  --checkpoint.folder checkpoint/rae_openimages_v3 \
  --validator.freq 500 \
  --validator.steps 16
