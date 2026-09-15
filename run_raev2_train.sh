
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
# 96k static U-ViT recipe. Token budgets, accumulation (x32 per step),
# validator cadence, and GAN/LR schedules all come from the config registry;
# this script only carries run identity (epochs, step cap, checkpoint folder).
# TOKENS_PER_STEP must match the registry: 97280 packed budget x 32.
TOKENS_PER_STEP=3112960
# Training stops after N_EPOCHS full passes over the dataset (--epochs).
# TOKENS_PER_EPOCH is only an estimate that sizes the step-based LR/GAN
# schedule horizon and the steps safety cap; the measured tokens per epoch
# are logged as rae/tokens_last_epoch so the estimate can be refined.
# Measured: full census of all 1,743,041 train images (JPEG-header dimensions
# only, no decode) through the recipe's Qwen smart_resize geometry on
# 2026-09-09: 1,742,424 valid images, 618 unreadable headers, mean 727.0
# post-merge tokens/image, p5/p50/p95 = 576/704/992.
TOKENS_PER_EPOCH=1266675661
RAE_CONFIG=${RAE_CONFIG:-rae_stage1_openimages_static_96k_uvit}

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
  --training.steps "${TRAINING_STEPS}" \
  --checkpoint.folder checkpoint/rae_openimages_$(date +%Y%m%d-%H%M%S)
