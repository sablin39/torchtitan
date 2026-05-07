#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

exec > >(tee -a output.log) 2>&1

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"

resolve_cmd() {
    local name="$1"
    local fallback="$2"
    if command -v "${name}" >/dev/null 2>&1; then
        command -v "${name}"
    elif [[ -x "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
    else
        echo "Could not find ${name}; activate .venv or install it." >&2
        return 1
    fi
}

python_cmd="$(resolve_cmd python "${repo_root}/.venv/bin/python")"
torchrun_cmd="$(resolve_cmd torchrun "${repo_root}/.venv/bin/torchrun")"

# Pipeline:
#   1. Export RWKV .pth + Qwen3.5 vision weights to an HF RWKV-VL checkpoint.
#   2. Convert that HF checkpoint to TorchTitan DCP.
#   3. Train with TorchTitan.
#   4. Convert the final TorchTitan DCP checkpoint back to HF and copy HF assets.
#
# Activate the environment before running:
#   source .venv/bin/activate
#
# WandB logging uses TorchTitan's built-in logger. Configure it with normal
# environment variables if needed:
#   export WANDB_PROJECT=torchtitan
#   export WANDB_RUN_NAME=rwkv-vl-train
#   export WANDB_MODE=offline  # optional, for offline/local logging
#
# Edit this block directly for now. We will replace it with a smarter config
# system later.

rwkv7_path="/mnt/raid0_8t/rwkv7-g1/rwkv7-g1d-0.4b-20260210-ctx8192.pth"
vision_model="/home/rwkv/models/Qwen3.5-0.8B"
dataset_path="/mnt/raid0_8t/LLaVA-OneVision-Data/chartqa(cauldron,llava_format)"

split="train"
ngpu="4"
seq_len="4096"
batch_size="32"
# Sequence packing is controlled by the multimodal dataloader, not by CP.
# packing_buffer_size is the number of tokenized samples kept in a CPU-side
# buffer before greedily combining them into seq_len rows. Larger values usually
# improve non-padding token occupancy, but increase preprocessing latency and
# host memory use. Set to "0" to disable packing and pad each sample normally.
# This is not batch size; batch_size still controls the number of packed rows
# per step/CP group.
packing_buffer_size="64"
# Set to an integer for a fixed-step run, or "epoch" to run until the finite
# dataloader is exhausted. With sequence packing, exact epoch steps are not known
# until samples are filtered, resized, tokenized, and packed.
steps="epoch"
max_epoch_steps="1000000000"
precision="bfloat16"
export_dtype="bfloat16"
model_name="rwkv_vl"
model_flavor="0.4B"
train_config="rwkv_vl_0_4b_chat"
# Comma-separated roots to train. Valid entries are vision_encoder, proj, llm,
# lm_head, and all. The normal llm selector includes the top-level lm_head.
# For a common projector+LM finetune with frozen vision, use: "proj,llm".
train_module="proj,llm"
projector_seed="1234"
activation_checkpoint_mode="none"
log_freq="1"
wandb="1"
overwrite="0"
optimizer_name="Adam"
learning_rate="1e-5"
lr_warmup_steps="500"
# Leave empty to use training_steps. In steps="epoch" mode, set this manually
# if you want the cosine decay horizon to be shorter than max_epoch_steps.
lr_total_steps=""
lr_decay_type="cosine"
lr_min_factor="0.1"
image_processor=""
min_pixels="65536"
max_pixels="2097152"
# 0 means no image-count cap. max_pixels is a shared per-sample pixel budget
# across all images in one chat example; set a positive image cap only as an
# emergency batch-memory guard.
max_images_per_batch="0"
max_position_embeddings=""
max_shard_size="1000GB"
output_root="${repo_root}/outputs/rwkv_vl_train_${timestamp}"

train_extra_args=(
    # Add extra torchtitan.train args here, for example:
    --parallelism.context-parallel-degree "${ngpu}"
    --parallelism.context-parallel-load-balancer None
    --compile.enable
    # --compile.components model
)

if [[ $# -gt 0 ]]; then
    echo "This script is configured by editing run_vl_train.sh directly." >&2
    echo "Command-line arguments are intentionally disabled for now." >&2
    exit 2
fi

if [[ -z "${rwkv7_path}" || -z "${vision_model}" || -z "${dataset_path}" ]]; then
    echo "Set rwkv7_path, vision_model, and dataset_path in run_vl_train.sh." >&2
    exit 2
fi

if [[ ! -f "${rwkv7_path}" ]]; then
    echo "RWKV checkpoint does not exist or is not a file: ${rwkv7_path}" >&2
    exit 1
fi

if [[ ! -e "${vision_model}" ]]; then
    echo "Warning: vision model is not a local path; assuming HF can resolve it: ${vision_model}" >&2
fi

if ! [[ "${ngpu}" =~ ^[0-9]+$ ]] || (( ngpu < 1 )); then
    echo "ngpu must be a positive integer, got: ${ngpu}" >&2
    exit 2
fi

if ! [[ "${batch_size}" =~ ^[0-9]+$ ]] || (( batch_size < 1 )); then
    echo "batch_size must be a positive integer, got: ${batch_size}" >&2
    exit 2
fi

if ! [[ "${seq_len}" =~ ^[0-9]+$ ]] || (( seq_len < 1 )); then
    echo "seq_len must be a positive integer, got: ${seq_len}" >&2
    exit 2
fi

if ! [[ "${packing_buffer_size}" =~ ^[0-9]+$ ]]; then
    echo "packing_buffer_size must be a non-negative integer, got: ${packing_buffer_size}" >&2
    exit 2
fi

run_until_epoch="0"
if [[ "${steps}" == "epoch" || "${steps}" == "auto" ]]; then
    run_until_epoch="1"
    training_steps="${max_epoch_steps}"
elif [[ "${steps}" =~ ^[0-9]+$ ]] && (( steps > 0 )); then
    training_steps="${steps}"
else
    echo "steps must be a positive integer, \"epoch\", or \"auto\"; got: ${steps}" >&2
    exit 2
fi

total_local_tokens=$((batch_size * seq_len))
if (( total_local_tokens % ngpu != 0 )); then
    echo "RWKV/FLA CP requires batch_size * seq_len to be divisible by ngpu." >&2
    echo "Got batch_size=${batch_size}, seq_len=${seq_len}, ngpu=${ngpu}." >&2
    exit 2
fi

hf_dir="${output_root}/hf_export"
dcp_dir="${output_root}/dcp_from_hf"
train_dump_dir="${output_root}/train"
final_hf_dir="${output_root}/hf_final"

if [[ "${overwrite}" == "1" ]]; then
    rm -rf "${hf_dir}" "${dcp_dir}" "${train_dump_dir}" "${final_hf_dir}"
fi

for path in "${hf_dir}" "${dcp_dir}" "${train_dump_dir}" "${final_hf_dir}"; do
    if [[ -e "${path}" ]]; then
        echo "Refusing to overwrite existing path: ${path}" >&2
        echo "Set overwrite=1 or choose a new output_root." >&2
        exit 1
    fi
done

mkdir -p "${output_root}"

echo "Artifacts:"
echo "  HF export:     ${hf_dir}"
echo "  DCP export:    ${dcp_dir}"
echo "  Train dump:    ${train_dump_dir}"
echo "  Final HF:      ${final_hf_dir}"

export_args=(
    "${repo_root}/scripts/rwkv7_exporter/export_hf_model.py"
    --rwkv7 "${rwkv7_path}"
    --vision-model "${vision_model}"
    --output "${hf_dir}"
    --multimodal
    --precision "${precision}"
    --max-shard-size "${max_shard_size}"
)

if [[ -n "${projector_seed}" ]]; then
    export_args+=(--projector-seed "${projector_seed}")
fi
if [[ -n "${image_processor}" ]]; then
    export_args+=(--image-processor "${image_processor}")
fi
if [[ -n "${max_pixels}" ]]; then
    export_args+=(--max-pixels "${max_pixels}")
fi
if [[ -n "${max_position_embeddings}" ]]; then
    export_args+=(--max-position-embeddings "${max_position_embeddings}")
fi

echo
echo "==> Step 1/4: Exporting RWKV-VL HF checkpoint"
"${python_cmd}" "${export_args[@]}"

echo
echo "==> Step 2/4: Converting HF checkpoint to DCP"
"${python_cmd}" "${repo_root}/scripts/checkpoint_conversion/convert_from_hf.py" \
    "${hf_dir}" \
    "${dcp_dir}" \
    --model_name "${model_name}" \
    --model_flavor "${model_flavor}"

train_args=(
    -m torchtitan.train
    --module "${model_name}"
    --config "${train_config}"
    --train-module "${train_module}"
    --hf-assets-path "${hf_dir}"
    --dump-folder "${train_dump_dir}"
    --metrics.log-freq "${log_freq}"
    --dataloader.dataset-path "${dataset_path}"
    --dataloader.split "${split}"
    --optimizer.name "${optimizer_name}"
    --optimizer.lr "${learning_rate}"
    --lr-scheduler.warmup-steps "${lr_warmup_steps}"
    --lr-scheduler.decay-type "${lr_decay_type}"
    --lr-scheduler.min-lr-factor "${lr_min_factor}"
    --training.seq-len "${seq_len}"
    --training.steps "${training_steps}"
    --training.local-batch-size "${batch_size}"
    --dataloader.packing-buffer-size "${packing_buffer_size}"
    --activation-checkpoint.mode "${activation_checkpoint_mode}"
    --checkpoint.enable
    --checkpoint.initial-load-path "${dcp_dir}"
    --checkpoint.interval "${training_steps}"
    --checkpoint.keep-latest-k 0
    --checkpoint.export-dtype "${export_dtype}"
)

if [[ "${run_until_epoch}" == "1" ]]; then
    train_args+=(--dataloader.no-infinite)
fi
if [[ -n "${lr_total_steps}" ]]; then
    train_args+=(--lr-scheduler.total-steps "${lr_total_steps}")
fi
if [[ "${wandb}" == "1" ]]; then
    train_args+=(--metrics.enable-wandb)
fi
if [[ -n "${min_pixels}" ]]; then
    train_args+=(--dataloader.min-pixels "${min_pixels}")
fi
if [[ -n "${max_pixels}" ]]; then
    train_args+=(--dataloader.max-pixels "${max_pixels}")
fi
if [[ -n "${max_images_per_batch}" ]]; then
    train_args+=(--dataloader.max-images-per-batch "${max_images_per_batch}")
fi
train_args+=("${train_extra_args[@]}")

echo
echo "==> Step 3/4: Training"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
"${torchrun_cmd}" \
    --standalone \
    --nproc-per-node="${ngpu}" \
    --local-ranks-filter="${LOG_RANK:-0}" \
    --role rank \
    --tee 3 \
    "${train_args[@]}"

if [[ "${run_until_epoch}" == "1" ]]; then
    trained_dcp_dir=""
    if [[ -d "${train_dump_dir}/checkpoint" ]]; then
        trained_dcp_dir="$(
            find "${train_dump_dir}/checkpoint" -mindepth 1 -maxdepth 1 -type d -name 'step-*' \
                | sed -E 's#^(.*step-)([0-9]+)$#\2\t&#' \
                | sort -n \
                | tail -1 \
                | cut -f2-
        )"
    fi
else
    trained_dcp_dir="${train_dump_dir}/checkpoint/step-${training_steps}"
fi
if [[ ! -d "${trained_dcp_dir}" ]]; then
    echo "Expected final DCP checkpoint not found: ${trained_dcp_dir}" >&2
    echo "Training may have ended before a checkpoint was saved. Check ${train_dump_dir}." >&2
    exit 1
fi
echo "Using trained DCP checkpoint: ${trained_dcp_dir}"

echo
echo "==> Step 4/4: Converting trained DCP checkpoint back to HF"
"${python_cmd}" "${repo_root}/scripts/checkpoint_conversion/convert_to_hf.py" \
    "${trained_dcp_dir}" \
    "${final_hf_dir}" \
    --hf_assets_path "${hf_dir}" \
    --model_name "${model_name}" \
    --model_flavor "${model_flavor}" \
    --export_dtype "${export_dtype}"

echo
echo "==> Copying HF remote-code/tokenizer/processor assets"
while IFS= read -r -d '' asset; do
    base="$(basename "${asset}")"
    case "${base}" in
        *.safetensors|model.safetensors.index.json|pytorch_model*.bin)
            continue
            ;;
    esac
    cp -a "${asset}" "${final_hf_dir}/${base}"
done < <(find "${hf_dir}" -mindepth 1 -maxdepth 1 -print0)

echo
echo "Done."
echo "  Initial HF checkpoint: ${hf_dir}"
echo "  Initial DCP checkpoint: ${dcp_dir}"
echo "  Training outputs:       ${train_dump_dir}"
echo "  Final HF checkpoint:    ${final_hf_dir}"
