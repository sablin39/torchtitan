#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Terminal tee is enabled after output directories are computed so the full
# stdout/stderr stream lands in the run artifacts.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"

python_cmd="python"
torchrun_cmd="torchrun"

# Pipeline:
#   1. Export RWKV .pth + Qwen3-VL vision weights to an HF RWKV-VL checkpoint.
#   2. Convert that HF checkpoint to TorchTitan DCP.
#   3. Train with TorchTitan.
#   4. Convert the final TorchTitan DCP checkpoint back to HF and copy HF assets.
#
# Setting resume_dcp_path below skips steps 1-2 and initializes training from
# an existing DCP checkpoint instead (weights only; the run starts at step 0).
#
# Activate the environment before running:
#   source .venv/bin/activate
#
# Experiment tracking uses TorchTitan's built-in logger. SwanLab runs as a
# backup mirror of W&B (via swanlab.sync_wandb) when the remote W&B connection
# is unstable, so the single tracking="1" switch below enables both backends
# with the same run name. SwanLab inherits the run name from W&B, so only the
# W&B env vars need to be configured:
#   export WANDB_PROJECT=torchtitan
#   export WANDB_RUN_NAME=rwkv-vl-train  # defaults to ${train_config}_${timestamp}
#   export WANDB_MODE=offline  # optional, for offline/local logging
#
# Edit this block directly for now. We will replace it with a smarter config
# system later.

# --- Inputs: checkpoints and datasets ---------------------------------------
rwkv7_path="${HOME}/models/rwkv7-g1/rwkv7-g1f-1.5b-20260419-ctx8192.pth"
vision_model="${HOME}/models/Qwen3-VL-2B-Instruct"
# W&B remote path: /data/HuggingFaceM4_FineVisionMax
dataset_path="${HOME}/data/LLaVA-OneVision-Data/ai2d(cauldron,llava_format)"
# Optional text-only SFT source. Image-text rows stay LLaVA/common-schema
# friendly; text rows are loaded through the strict Nemotron chat processor.
text_dataset_path="${HOME}/data/nemotron_cleaned/agentic_v2_cleaned"
text_split="train"
text_sample_probability="0.5"
split="train"

# --- Fine-tune from an existing DCP checkpoint -------------------------------
# Empty (default): run the full pipeline — export the HF seed checkpoint,
# convert it to DCP, and train from that seed. Set to a step-N DCP checkpoint
# directory from a previous run (e.g.
# outputs/rwkv_vl_train_YYYYmmdd_HHMMSS/train/checkpoint/step-2000) to skip the
# export/convert steps and initialize training from those weights instead.
# This starts a fresh run at step 0: only model weights are loaded
# (checkpoint.initial_load_model_only keeps its default), so the optimizer,
# LR schedule, and all training configs follow this script, not the checkpoint.
# The train_config and projector_* knobs must match the checkpoint being loaded.
resume_dcp_path=""
# HF assets (tokenizer/processor/config/remote code) are still needed for
# training and for the final HF export. Empty means: reuse the hf_export dir
# of the run that produced resume_dcp_path (<run_root>/hf_export). Set this to
# use HF assets from somewhere else. Ignored when resume_dcp_path is empty.
resume_hf_assets=""

# --- Model ----------------------------------------------------------------------
# train_config picks the model flavor from the registry
# (torchtitan/models/rwkv_vl/config_registry.py); the checkpoint/vision pairs
# each config was validated with:
#   1.5B-v100M:
#     rwkv7_path="/home/molin/models/rwkv7-g1/rwkv7-g1f-1.5b-20260419-ctx8192.pth"
#     vision_model="/home/molin/models/Qwen3.5-0.8B"
#     train_config="rwkv_vl_1_5b_v100m_chat"
#   1.5B-v400M:
#     rwkv7_path="/home/molin/models/rwkv7-g1/rwkv7-g1f-1.5b-20260419-ctx8192.pth"
#     vision_model="/home/molin/models/Qwen3-VL-2B-Instruct"
#     train_config="rwkv_vl_1_5b_v400m_chat"
train_config="rwkv_vl_1_5b_v400m_chat"
# RWKV7 DPLR chunk size for the language backbone. The model default is 64;
# local long-sequence sweeps favored 32 for packed CP training.
backbone_chunk_size="64"

# --- Visual projector ----------------------------------------------------------
# "cross_attn" uses TokenPacker seed queries followed by masked retrieval from
# raw ViT DeepStack memories at selected RWKV layers. "mlp" keeps the additive
# projected-DeepStack variant.
projector_kind="cross_attn"
# Norm used inside the projector ("layernorm" or "rmsnorm").
projector_norm="layernorm"
# Inner-FFN activation for the MLP projector ("relu" / "gelu" / "swiglu").
# The cross-attention projector does not use an FFN.
projector_ffn="relu"
# Comma-separated post-block ViT memories and corresponding RWKV injection
# layers. Empty uses the selected flavor defaults: 100M -> 2,6,9 and 400M ->
# 5,11,17; both 24-layer RWKV flavors inject after layers 2,9,16.
projector_visual_layer_indices=""
projector_language_layer_indices=""
# GQA layout. Empty derives the flavor defaults from the ViT: 12Q/6KV for the
# 100M encoder and 16Q/8KV for the 400M encoder. This 2:1 GQA layout doubles
# K/V capacity over the previous 4:1 setting without widening Q attention.
projector_num_query_heads=""
projector_num_key_value_heads=""
# 1 shares RWKV-Q and visual-K/V/output projections across all retrieval depths
# and the TokenPacker seed K/V/O path. 0 gives every depth its own Q/K/V/O set
# and gives the seed path its own K/V/O set.
tie_projector_qkvo="1"
# TokenPacker query-grid downsampling ratio. The processor inserts image tokens
# at vision_spatial_merge_size * this ratio while the frozen ViT keeps its native
# patch order. Only supported with projector_kind=cross_attn when greater than 1.
projector_extra_merge_size="2"
# Static FlexAttention buckets for the cross_attn projector (cross_attn only).
# Each forward pads Q and K/V to exactly these sizes so the compiled FlexAttention
# kernel sees a single static shape. Pick values >= the largest expected per-batch
# image_pad / vision token counts. Leave empty to use the dynamic ladder (faster
# kernels but may re-compile per shape).
projector_q_bucket=""
projector_kv_bucket=""
# Seed for the freshly initialized projector weights at HF export time
# (distinct from project_seed, which seeds training itself).
projector_seed="1234"

# --- Training schedule ---------------------------------------------------------
# Set to an integer for a fixed-step run, or "epoch" to run until the finite
# dataloader is exhausted. With sequence packing, exact epoch steps are not known
# until samples are filtered, resized, tokenized, and packed.
steps="epoch"
max_epoch_steps="1000000000"
seq_len="8192"
# The recovered 4096-token W&B run used batch_size=24. For 8192-token local
# stress on this shared 4x96GB workstation, batch_size=8 is the verified default.
batch_size="8"
# batch_size is TorchTitan training.local_batch_size. With RWKV/FLA CP it is
# the number of packed seq_len rows per batch-parallel group. CP shards the
# flattened tokens inside each row group; it does not multiply batch size.
# The effective global batch is:
#   batch_size * (ngpu / context_parallel_degree) * gradient_accumulation_steps
# which the script derives and passes to TorchTitan as training.global-batch-size
# (TorchTitan has no dedicated accumulation field; it computes the step count from
# global_batch_size / (local_batch_size * batch_degree)).
# Sequence packing is controlled by the multimodal dataloader, not by CP.
# Gradient accumulation: how many forward/backward microbatches are summed before
# each optimizer step. Default 1 keeps the pre-accumulation behavior. The trainer
# buffers all microbatches on the CPU before stepping, so large values with
# multimodal data raise host RAM use.
gradient_accumulation_steps="1"
# packing_buffer_size is the number of tokenized samples kept in a CPU-side
# buffer before greedily combining them into seq_len rows. Larger values usually
# improve non-padding token occupancy, but increase preprocessing latency and
# host memory use. Set to "0" to disable packing and pad each sample normally.
packing_buffer_size="64"

# --- Learning rates --------------------------------------------------------------
# Per-root learning rates. A value of 0 freezes that root and skips selective
# FSDP sharding for it. lm_head always follows llm_lr.
vision_encoder_lr="0"
proj_lr="1e-4"
llm_lr="1e-5"
# --optimizer.lr has no separate knob: it is only the scaling base for the
# per-root param groups (effective LR = base * root_lr / base == root_lr), so
# any positive value works. Derive it as the largest root LR so it stays
# positive whenever at least one root is trainable.
optimizer_base_lr="$(awk -v a="${vision_encoder_lr}" -v b="${proj_lr}" -v c="${llm_lr}" 'BEGIN{ m=a; if (b>m) m=b; if (c>m) m=c; print m }')"
optimizer_name="Adam"
weight_decay="0"
lr_warmup_steps="2000"
# Leave empty to use training_steps. In steps="epoch" mode, set this manually
# if you want the cosine decay horizon to be shorter than max_epoch_steps.
lr_total_steps=""
lr_decay_type="linear"
lr_min_factor="1.0"
# Seed for training itself (--debug.seed; distinct from projector_seed).
project_seed="1234"

# --- Checkpointing -----------------------------------------------------------------
checkpoint_interval="2000"
checkpoint_keep_latest_k="0"
overwrite="0"

# --- Hardware and parallelism --------------------------------------------------------
ngpu="4"
# Size of each context-parallel group. For context_parallel_degree=2, ngpu=4
# creates two CP groups of 2 ranks; ngpu=8 creates four CP groups of 2 ranks.
context_parallel_degree="1"
# Data parallel layout. The default keeps current FSDP behavior by sharding
# across all non-CP ranks. For replicated DP on 8 GPUs with CP=1, set:
#   data_parallel_replicate_degree="8"
#   data_parallel_shard_degree="1"
# With CP>1, use data_parallel_replicate_degree=ngpu/context_parallel_degree.
data_parallel_replicate_degree="1"
data_parallel_shard_degree="-1"
activation_checkpoint_mode="full"
# RWKV-VL selective activation checkpointing is usable with CP on and off when
# using BF16 model construction, compile, and normal FSDP.
# torchrun warns about OMP_NUM_THREADS because every rank and dataloader worker
# can otherwise spawn a large CPU thread pool. Start at 1 for multimodal CP; if
# RAM and CPU load look stable, benchmark 2. A rough upper bound is:
# physical_cores / (ngpu * (1 + dataloader_num_workers)).
omp_num_threads="1"
# Correct CUDA allocator knob. The older PYTORCH_ALLOC_CONF name is ignored by
# PyTorch for CUDA memory management.
pytorch_cuda_alloc_conf="expandable_segments:True"

# --- Dataloader ----------------------------------------------------------------------
# Conservative dataloader overlap for multimodal packing. Each worker can hold
# prefetched packed batches containing many resized images, so keep this small
# on RAM-constrained machines. With CP, this is per rank.
dataloader_num_workers="0"
dataloader_persistent_workers="1"
dataloader_prefetch_factor="1"
dataloader_pin_memory="0"
# Store preprocessed visual patch tensors in this dtype before worker IPC and
# H2D transfer. For BF16 training this roughly halves pixel_values host memory
# and transfer volume compared with float32 while resize/normalize still runs
# in float32 inside the processor.
dataloader_pixel_values_dtype="bfloat16"
min_pixels="65536"
max_pixels="3145728"
# 0 means no image-count cap. max_pixels is a shared per-sample pixel budget
# across all images in one chat example; set a positive image cap only as an
# emergency batch-memory guard.
max_images_per_batch="0"
# Flat ViT patch bucketing stabilizes FlexAttention sequence shapes for image
# patch streams. 0 disables bucketing and preserves the exact old data path.
# Useful benchmark sweep values: 0, 16384, 32768, 65536.
# For a bucket sweep, edit vit_patch_bucket_size and keep
# torchinductor_cache_dir distinct for each cold-cache run.
vit_patch_bucket_size="32768"

# --- Logging and debugging -------------------------------------------------------------
# Enable W&B + SwanLab together. SwanLab is configured as the backup mirror so
# runs survive transient W&B connectivity issues.
tracking="1"
nvml_metrics="1"
log_freq="1"
# Set to 1 for crash/numerics/compiler debugging. This enables heavyweight
# asserts and verbose logs. Leave 0 for speed-equivalent runs. FlexAttention
# autotune stays enabled by the model config either way.
debugging="0"
# Set to "auto" to capture the full shell/torchrun terminal stream in the train
# artifact directory. Set empty to disable shell-level tee logging.
terminal_log_file="auto"
# FlexAttention autotune choices are useful in both speed and debug runs.
flex_attention_log_file="auto"

# --- Advanced (rarely touched) ------------------------------------------------------------
model_name="rwkv_vl"
# Weight dtype of the initial HF export (step 1).
precision="bfloat16"
# Weight dtype of the trained HF export (step 4) and the final DCP checkpoint.
export_dtype="bfloat16"
# Optional image processor source for the exported HF checkpoint. Empty means:
# inherit the preprocessor from vision_model. Training does not read it.
image_processor=""
max_position_embeddings=""
max_shard_size="1000GB"
# Keep Inductor caches separate across bucket-size sweeps when benchmarking
# cold compile/autotune behavior. Leave empty to let PyTorch choose the cache.
torchinductor_cache_dir="/tmp/tt_vit_bucket_${vit_patch_bucket_size}_cp${context_parallel_degree}_bs${batch_size}"

# Derived run identity (not config knobs).
output_root="${repo_root}/outputs/rwkv_vl_train_${timestamp}"
tracking_run_name="${WANDB_RUN_NAME:-${train_config}_${timestamp}}"

train_extra_args=(
    # Add extra torchtitan.train args here, for example:
    --parallelism.data-parallel-replicate-degree "${data_parallel_replicate_degree}"
    --parallelism.data-parallel-shard-degree "${data_parallel_shard_degree}"
    --parallelism.context-parallel-degree "${context_parallel_degree}"
    --parallelism.context-parallel-load-balancer None
    --compile.enable
    # --compile.components model
)

if [[ $# -gt 0 ]]; then
    echo "This script is configured by editing run_vl_train.sh directly." >&2
    echo "Command-line arguments are intentionally disabled for now." >&2
    exit 2
fi

if [[ -z "${dataset_path}" ]]; then
    echo "Set dataset_path in run_vl_train.sh." >&2
    exit 2
fi

if [[ -n "${resume_dcp_path}" ]]; then
    if [[ ! -d "${resume_dcp_path}" ]]; then
        echo "resume_dcp_path does not exist or is not a directory: ${resume_dcp_path}" >&2
        exit 1
    fi
else
    if [[ -z "${rwkv7_path}" || -z "${vision_model}" ]]; then
        echo "Set rwkv7_path and vision_model in run_vl_train.sh." >&2
        exit 2
    fi
    if [[ ! -f "${rwkv7_path}" ]]; then
        echo "RWKV checkpoint does not exist or is not a file: ${rwkv7_path}" >&2
        exit 1
    fi
    if [[ ! -e "${vision_model}" ]]; then
        echo "Warning: vision model is not a local path; assuming HF can resolve it: ${vision_model}" >&2
    fi
fi

if ! [[ "${ngpu}" =~ ^[0-9]+$ ]] || (( ngpu < 1 )); then
    echo "ngpu must be a positive integer, got: ${ngpu}" >&2
    exit 2
fi

if ! [[ "${context_parallel_degree}" =~ ^[0-9]+$ ]] || (( context_parallel_degree < 1 )); then
    echo "context_parallel_degree must be a positive integer, got: ${context_parallel_degree}" >&2
    exit 2
fi

if (( ngpu % context_parallel_degree != 0 )); then
    echo "ngpu must be divisible by context_parallel_degree." >&2
    echo "Got ngpu=${ngpu}, context_parallel_degree=${context_parallel_degree}." >&2
    exit 2
fi

if ! [[ "${data_parallel_replicate_degree}" =~ ^[0-9]+$ ]] || (( data_parallel_replicate_degree < 1 )); then
    echo "data_parallel_replicate_degree must be a positive integer, got: ${data_parallel_replicate_degree}" >&2
    exit 2
fi
if ! [[ "${data_parallel_shard_degree}" =~ ^-?[0-9]+$ ]] || (( data_parallel_shard_degree == 0 || data_parallel_shard_degree < -1 )); then
    echo "data_parallel_shard_degree must be -1 or a positive integer, got: ${data_parallel_shard_degree}" >&2
    exit 2
fi
if (( data_parallel_shard_degree == -1 )); then
    dp_base=$((data_parallel_replicate_degree * context_parallel_degree))
    if (( ngpu % dp_base != 0 )); then
        echo "ngpu must be divisible by data_parallel_replicate_degree * context_parallel_degree when data_parallel_shard_degree=-1." >&2
        echo "Got ngpu=${ngpu}, data_parallel_replicate_degree=${data_parallel_replicate_degree}, context_parallel_degree=${context_parallel_degree}." >&2
        exit 2
    fi
    effective_data_parallel_shard_degree=$((ngpu / dp_base))
else
    effective_data_parallel_shard_degree="${data_parallel_shard_degree}"
    if (( data_parallel_replicate_degree * data_parallel_shard_degree * context_parallel_degree != ngpu )); then
        echo "Invalid data/context parallel layout." >&2
        echo "Expected data_parallel_replicate_degree * data_parallel_shard_degree * context_parallel_degree == ngpu." >&2
        echo "Got ${data_parallel_replicate_degree} * ${data_parallel_shard_degree} * ${context_parallel_degree} != ${ngpu}." >&2
        exit 2
    fi
fi

if ! [[ "${batch_size}" =~ ^[0-9]+$ ]] || (( batch_size < 1 )); then
    echo "batch_size must be a positive integer, got: ${batch_size}" >&2
    exit 2
fi

if ! [[ "${gradient_accumulation_steps}" =~ ^[0-9]+$ ]] || (( gradient_accumulation_steps < 1 )); then
    echo "gradient_accumulation_steps must be a positive integer, got: ${gradient_accumulation_steps}" >&2
    exit 2
fi

if ! [[ "${seq_len}" =~ ^[0-9]+$ ]] || (( seq_len < 1 )); then
    echo "seq_len must be a positive integer, got: ${seq_len}" >&2
    exit 2
fi

if ! [[ "${backbone_chunk_size}" =~ ^[0-9]+$ ]] || (( backbone_chunk_size < 16 )); then
    echo "backbone_chunk_size must be an integer >= 16, got: ${backbone_chunk_size}" >&2
    exit 2
fi
if (( backbone_chunk_size & (backbone_chunk_size - 1) )); then
    echo "backbone_chunk_size must be a power of two, got: ${backbone_chunk_size}" >&2
    exit 2
fi

if ! [[ "${packing_buffer_size}" =~ ^[0-9]+$ ]]; then
    echo "packing_buffer_size must be a non-negative integer, got: ${packing_buffer_size}" >&2
    exit 2
fi
if ! [[ "${vit_patch_bucket_size}" =~ ^[0-9]+$ ]]; then
    echo "vit_patch_bucket_size must be a non-negative integer, got: ${vit_patch_bucket_size}" >&2
    exit 2
fi

require_bool() {
    local name="$1"
    local value="${!name}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1, got: ${value}" >&2
        exit 2
    fi
}

for bool_name in \
    debugging \
    tie_projector_qkvo; do
    require_bool "${bool_name}"
done

# Internal diagnostics/env defaults derived from the single debugging switch.
torch_logs=""
python_faulthandler="0"
triton_debug="0"
torch_show_cpp_stacktraces="0"
torch_disable_addr2line="0"
torch_cpp_log_level=""
torch_distributed_debug=""
torchinductor_nan_asserts="0"
torchinductor_runtime_triton_nan_asserts="0"
nccl_debug="WARN"
nccl_debug_subsys=""
nccl_debug_file=""
torch_nccl_async_error_handling="1"
torch_nccl_enable_monitoring="1"
torch_nccl_heartbeat_timeout_sec="600"
torch_nccl_wait_timeout_dump_milsec="120000"
torch_nccl_log_cpp_stack_on_unclean_shutdown="0"
torch_nccl_flight_recorder="0"
torch_nccl_trace_buffer_size="8192"
torch_nccl_trace_cpp_stack="0"
torch_nccl_desync_debug="0"
torch_nccl_enable_timing="0"
torch_nccl_nan_check="0"

if [[ "${debugging}" == "1" ]]; then
    torch_logs="+inductor,recompiles,graph_breaks"
    python_faulthandler="1"
    triton_debug="1"
    torch_show_cpp_stacktraces="1"
    torch_disable_addr2line="1"
    torch_cpp_log_level="INFO"
    torch_distributed_debug="DETAIL"
    torchinductor_nan_asserts="1"
    torchinductor_runtime_triton_nan_asserts="1"
    torch_nccl_log_cpp_stack_on_unclean_shutdown="1"
    torch_nccl_flight_recorder="1"
    torch_nccl_trace_cpp_stack="1"
    torch_nccl_desync_debug="1"
    torch_nccl_enable_timing="1"
    torch_nccl_nan_check="1"
fi

for bool_name in \
    python_faulthandler \
    triton_debug \
    torch_show_cpp_stacktraces \
    torch_disable_addr2line \
    torchinductor_nan_asserts \
    torchinductor_runtime_triton_nan_asserts \
    torch_nccl_flight_recorder \
    torch_nccl_trace_cpp_stack \
    torch_nccl_desync_debug \
    torch_nccl_enable_timing \
    torch_nccl_nan_check; do
    require_bool "${bool_name}"
done

if ! [[ "${torch_nccl_trace_buffer_size}" =~ ^[0-9]+$ ]]; then
    echo "torch_nccl_trace_buffer_size must be a non-negative integer, got: ${torch_nccl_trace_buffer_size}" >&2
    exit 2
fi
if ! [[ "${torch_nccl_heartbeat_timeout_sec}" =~ ^[0-9]+$ ]] || (( torch_nccl_heartbeat_timeout_sec < 1 )); then
    echo "torch_nccl_heartbeat_timeout_sec must be a positive integer, got: ${torch_nccl_heartbeat_timeout_sec}" >&2
    exit 2
fi
if ! [[ "${torch_nccl_wait_timeout_dump_milsec}" =~ ^[0-9]+$ ]]; then
    echo "torch_nccl_wait_timeout_dump_milsec must be a non-negative integer, got: ${torch_nccl_wait_timeout_dump_milsec}" >&2
    exit 2
fi

if ! [[ "${checkpoint_interval}" =~ ^[0-9]+$ ]] || (( checkpoint_interval < 1 )); then
    echo "checkpoint_interval must be a positive integer, got: ${checkpoint_interval}" >&2
    exit 2
fi

if ! [[ "${checkpoint_keep_latest_k}" =~ ^[0-9]+$ ]]; then
    echo "checkpoint_keep_latest_k must be a non-negative integer, got: ${checkpoint_keep_latest_k}" >&2
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
if (( total_local_tokens % context_parallel_degree != 0 )); then
    echo "RWKV/FLA CP requires batch_size * seq_len to be divisible by context_parallel_degree." >&2
    echo "Got batch_size=${batch_size}, seq_len=${seq_len}, context_parallel_degree=${context_parallel_degree}." >&2
    exit 2
fi

batch_parallel_degree=$((ngpu / context_parallel_degree))

# TorchTitan derives gradient accumulation as
#   global_batch_size / (local_batch_size * batch_degree),
# where batch_degree == batch_parallel_degree (DP replicate * shard; CP is
# excluded since it shards the sequence, not the batch). Derive global_batch_size
# here so the user-facing knob is the human-friendly step count, not a raw total.
global_batch_size=$((batch_size * batch_parallel_degree * gradient_accumulation_steps))

# model_flavor is implied by train_config: each training config is bound to
# exactly one flavor in torchtitan/models/rwkv_vl/config_registry.py. The
# HF <-> DCP converters take the flavor, so resolve it from the registry.
if ! model_flavor="$("${python_cmd}" -c "
import importlib
registry = importlib.import_module('torchtitan.models.${model_name}.config_registry')
print(getattr(registry, '${train_config}')().model_spec.flavor)
")"; then
    echo "Failed to resolve model flavor from train_config: ${train_config}" >&2
    exit 2
fi

if ! projector_defaults="$("${python_cmd}" -c "
import importlib
registry = importlib.import_module('torchtitan.models.${model_name}.config_registry')
model = getattr(registry, '${train_config}')().model_spec.model
print(
    ','.join(str(index) for index in model.vision_encoder.deepstack_visual_indices),
    ','.join(str(index) for index in model.proj.language_layer_indices),
    model.proj.num_query_heads,
    model.proj.num_key_value_heads,
    sep='\t',
)
")"; then
    echo "Failed to resolve projector defaults from train_config: ${train_config}" >&2
    exit 2
fi
IFS=$'\t' read -r \
    default_projector_visual_layers \
    default_projector_language_layers \
    default_projector_query_heads \
    default_projector_kv_heads <<< "${projector_defaults}"
projector_visual_layer_indices="${projector_visual_layer_indices:-${default_projector_visual_layers}}"
projector_language_layer_indices="${projector_language_layer_indices:-${default_projector_language_layers}}"
projector_num_query_heads="${projector_num_query_heads:-${default_projector_query_heads}}"
projector_num_key_value_heads="${projector_num_key_value_heads:-${default_projector_kv_heads}}"

IFS=',' read -r -a projector_visual_layers <<< "${projector_visual_layer_indices}"
IFS=',' read -r -a projector_language_layers <<< "${projector_language_layer_indices}"
if (( ${#projector_visual_layers[@]} != ${#projector_language_layers[@]} )); then
    echo "projector visual and language layer lists must have equal length." >&2
    exit 2
fi
for layer_index in "${projector_visual_layers[@]}" "${projector_language_layers[@]}"; do
    if ! [[ "${layer_index}" =~ ^[0-9]+$ ]]; then
        echo "projector layer indices must be non-negative integers, got: ${layer_index}" >&2
        exit 2
    fi
done
if ! [[ "${projector_num_query_heads}" =~ ^[0-9]+$ ]] || (( projector_num_query_heads < 1 )); then
    echo "projector_num_query_heads must be a positive integer." >&2
    exit 2
fi
if ! [[ "${projector_num_key_value_heads}" =~ ^[0-9]+$ ]] || (( projector_num_key_value_heads < 1 )); then
    echo "projector_num_key_value_heads must be a positive integer." >&2
    exit 2
fi
if (( projector_num_query_heads % projector_num_key_value_heads != 0 )); then
    echo "projector_num_query_heads must be divisible by projector_num_key_value_heads." >&2
    exit 2
fi

dcp_dir="${output_root}/dcp_from_hf"
if [[ -n "${resume_dcp_path}" ]]; then
    # Reuse the HF assets of the run that produced the checkpoint
    # (<run_root>/train/checkpoint/step-N -> <run_root>/hf_export) unless
    # resume_hf_assets overrides them.
    if [[ -n "${resume_hf_assets}" ]]; then
        hf_dir="${resume_hf_assets}"
    else
        hf_dir="$(realpath -m "$(dirname "$(dirname "$(dirname "${resume_dcp_path}")")")/hf_export")"
    fi
    if [[ ! -d "${hf_dir}" ]]; then
        echo "HF assets dir not found: ${hf_dir}" >&2
        echo "Set resume_hf_assets to an existing HF export directory." >&2
        exit 1
    fi
    initial_load_dcp="${resume_dcp_path}"
else
    hf_dir="${output_root}/hf_export"
    initial_load_dcp="${dcp_dir}"
fi
train_dump_dir="${output_root}/train"
final_hf_dir="${output_root}/hf_final"

if [[ "${flex_attention_log_file}" == "auto" ]]; then
    flex_attention_log_file="${output_root}/flex_attention_autotune"
fi

if [[ "${overwrite}" == "1" ]]; then
    rm -rf "${train_dump_dir}" "${final_hf_dir}"
    # In resume mode hf_dir is a pre-existing external assets dir and dcp_dir
    # is not created; never delete them.
    if [[ -z "${resume_dcp_path}" ]]; then
        rm -rf "${hf_dir}" "${dcp_dir}"
    fi
fi

guard_paths=("${train_dump_dir}" "${final_hf_dir}")
if [[ -z "${resume_dcp_path}" ]]; then
    guard_paths=("${hf_dir}" "${dcp_dir}" "${guard_paths[@]}")
fi
for path in "${guard_paths[@]}"; do
    if [[ -e "${path}" ]]; then
        echo "Refusing to overwrite existing path: ${path}" >&2
        echo "Set overwrite=1 or choose a new output_root." >&2
        exit 1
    fi
done

mkdir -p "${output_root}"
if [[ "${terminal_log_file}" == "auto" ]]; then
    terminal_log_file="${train_dump_dir}/terminal.log"
fi
if [[ -n "${terminal_log_file}" ]]; then
    mkdir -p "$(dirname "${terminal_log_file}")"
    if command -v stdbuf >/dev/null 2>&1; then
        exec > >(stdbuf -oL -eL tee -a "${terminal_log_file}") 2>&1
    else
        exec > >(tee -a "${terminal_log_file}") 2>&1
    fi
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    echo "Terminal log: ${terminal_log_file}"
fi
if [[ -n "${flex_attention_log_file}" ]]; then
    mkdir -p "$(dirname "${flex_attention_log_file}")"
fi
if [[ -n "${nccl_debug_file}" ]]; then
    mkdir -p "$(dirname "${nccl_debug_file}")"
fi

echo "Artifacts:"
if [[ -n "${resume_dcp_path}" ]]; then
    echo "  HF assets:     ${hf_dir} (reused)"
else
    echo "  HF export:     ${hf_dir}"
    echo "  DCP export:    ${dcp_dir}"
fi
echo "  Train dump:    ${train_dump_dir}"
echo "  Final HF:      ${final_hf_dir}"
if [[ -n "${resume_dcp_path}" ]]; then
    echo "Resume:"
    echo "  Init weights:  ${initial_load_dcp}"
    echo "  Mode:          model weights only; step 0, fresh optimizer/LR schedule"
fi
echo "Tracking:"
echo "  Enabled:       ${tracking} (W&B + SwanLab backup)"
echo "  Run name:      ${tracking_run_name}"
echo "Parallelism:"
echo "  GPUs:          ${ngpu}"
echo "  CP degree:     ${context_parallel_degree}"
echo "  DP replicate:  ${data_parallel_replicate_degree}"
echo "  DP shard:      ${data_parallel_shard_degree} (effective ${effective_data_parallel_shard_degree})"
echo "  Batch groups:  ${batch_parallel_degree}"
echo "  Grad accum:    ${gradient_accumulation_steps} step(s)/update"
echo "  Global batch:  ${global_batch_size} (local ${batch_size} x ${batch_parallel_degree} x ${gradient_accumulation_steps})"
echo "Bucketing:"
echo "  ViT patches:   ${vit_patch_bucket_size} (0 disables)"
echo "  Inductor dir:  ${torchinductor_cache_dir:-<torch default>}"
echo "  TORCH_LOGS:    ${torch_logs:-<unset>}"
echo "Projector:"
echo "  Kind:          ${projector_kind}"
echo "  ViT layers:    ${projector_visual_layer_indices}"
echo "  RWKV layers:   ${projector_language_layer_indices}"
echo "  Q/KV heads:    ${projector_num_query_heads}/${projector_num_key_value_heads}"
echo "  Tie QKVO:      ${tie_projector_qkvo}"
echo "  Extra merge:   ${projector_extra_merge_size}"
echo "Datasets:"
echo "  Image-text:    ${dataset_path}"
echo "  Text:          ${text_dataset_path:-<disabled>}"
echo "  Text split:    ${text_split}"
echo "  Text mix prob: ${text_sample_probability}"
echo "  Project seed:  ${project_seed:-<unset>}"
echo "Diagnostics:"
echo "  Debugging:     ${debugging}"
echo "  Terminal log:  ${terminal_log_file:-<unset>}"
echo "  Python faults: ${python_faulthandler}"
echo "  Triton debug:  ${triton_debug}"
echo "  C++ stacks:    ${torch_show_cpp_stacktraces}"
echo "  NaN asserts:   ${torchinductor_nan_asserts}/${torchinductor_runtime_triton_nan_asserts}"
echo "  addr2line:     $([[ "${torch_disable_addr2line}" == "1" ]] && echo disabled || echo enabled)"
echo "  FlexAttn log:  ${flex_attention_log_file:-<unset>}"
echo "  NCCL debug:    ${nccl_debug:-<unset>}"
echo "  NCCL flight:   ${torch_nccl_flight_recorder}"

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
if [[ -n "${projector_kind}" ]]; then
    export_args+=(--projector-kind "${projector_kind}")
fi
if [[ -n "${projector_norm}" ]]; then
    export_args+=(--projector-norm "${projector_norm}")
fi
if [[ -n "${projector_ffn}" ]]; then
    export_args+=(--projector-ffn "${projector_ffn}")
fi
export_args+=(--projector-visual-layer-indices "${projector_visual_layer_indices}")
export_args+=(--projector-language-layer-indices "${projector_language_layer_indices}")
export_args+=(--projector-num-query-heads "${projector_num_query_heads}")
export_args+=(--projector-num-key-value-heads "${projector_num_key_value_heads}")
if [[ "${tie_projector_qkvo}" == "1" ]]; then
    export_args+=(--tie-projector-qkvo)
else
    export_args+=(--no-tie-projector-qkvo)
fi
if [[ -n "${projector_extra_merge_size}" ]]; then
    export_args+=(--projector-extra-merge-size "${projector_extra_merge_size}")
fi

if [[ -z "${resume_dcp_path}" ]]; then
    echo
    echo "==> Step 1/4: Exporting RWKV-VL HF checkpoint"
    "${python_cmd}" "${export_args[@]}"
else
    echo
    echo "==> Steps 1-2/4: Skipped; initializing training from ${resume_dcp_path}"
fi

convert_proj_args=()
if [[ -n "${projector_kind}" ]]; then
    convert_proj_args+=(--projector_kind "${projector_kind}")
fi
if [[ -n "${projector_norm}" ]]; then
    convert_proj_args+=(--projector_norm "${projector_norm}")
fi
if [[ -n "${projector_ffn}" ]]; then
    convert_proj_args+=(--projector_ffn "${projector_ffn}")
fi
convert_proj_args+=(--projector_visual_layer_indices "${projector_visual_layers[@]}")
convert_proj_args+=(--projector_language_layer_indices "${projector_language_layers[@]}")
convert_proj_args+=(--projector_num_query_heads "${projector_num_query_heads}")
convert_proj_args+=(--projector_num_key_value_heads "${projector_num_key_value_heads}")
if [[ "${tie_projector_qkvo}" == "1" ]]; then
    convert_proj_args+=(--tie_projector_qkvo)
else
    convert_proj_args+=(--no-tie_projector_qkvo)
fi
if [[ -n "${projector_extra_merge_size}" ]]; then
    convert_proj_args+=(--projector_extra_merge_size "${projector_extra_merge_size}")
fi

if [[ -z "${resume_dcp_path}" ]]; then
    echo
    echo "==> Step 2/4: Converting HF checkpoint to DCP"
    "${python_cmd}" "${repo_root}/scripts/checkpoint_conversion/convert_from_hf.py" \
        "${hf_dir}" \
        "${dcp_dir}" \
        --model_name "${model_name}" \
        --model_flavor "${model_flavor}" \
        "${convert_proj_args[@]}"
fi

train_args=(
    -m torchtitan.train
    --module "${model_name}"
    --config "${train_config}"
    --hf-assets-path "${hf_dir}"
    --dump-folder "${train_dump_dir}"
    --metrics.log-freq "${log_freq}"
    --dataloader.dataset-path "${dataset_path}"
    --dataloader.split "${split}"
    --dataloader.text-sample-probability "${text_sample_probability}"
    --optimizer.name "${optimizer_name}"
    --optimizer.lr "${optimizer_base_lr}"
    --optimizer.weight-decay "${weight_decay}"
    --module-lrs.vision-encoder "${vision_encoder_lr}"
    --module-lrs.proj "${proj_lr}"
    --module-lrs.llm "${llm_lr}"
    --backbone-chunk-size "${backbone_chunk_size}"
    --lr-scheduler.warmup-steps "${lr_warmup_steps}"
    --lr-scheduler.decay-type "${lr_decay_type}"
    --lr-scheduler.min-lr-factor "${lr_min_factor}"
    --training.seq-len "${seq_len}"
    --training.steps "${training_steps}"
    --training.local-batch-size "${batch_size}"
    --training.global-batch-size "${global_batch_size}"
    --dataloader.packing-buffer-size "${packing_buffer_size}"
    --dataloader.vit-patch-bucket-size "${vit_patch_bucket_size}"
    --dataloader.num-workers "${dataloader_num_workers}"
    --dataloader.prefetch-factor "${dataloader_prefetch_factor}"
    --dataloader.pixel-values-dtype "${dataloader_pixel_values_dtype}"
    --activation-checkpoint.mode "${activation_checkpoint_mode}"
    --checkpoint.enable
    --checkpoint.initial-load-path "${initial_load_dcp}"
    --checkpoint.interval "${checkpoint_interval}"
    --checkpoint.keep-latest-k "${checkpoint_keep_latest_k}"
    --checkpoint.export-dtype "${export_dtype}"
)

if [[ -n "${text_dataset_path}" ]]; then
    train_args+=(--dataloader.text-dataset-path "${text_dataset_path}")
    train_args+=(--dataloader.text-split "${text_split}")
fi
if [[ -n "${project_seed}" ]]; then
    train_args+=(--debug.seed "${project_seed}")
fi
if [[ "${run_until_epoch}" == "1" ]]; then
    train_args+=(--dataloader.no-infinite)
fi
if [[ -n "${lr_total_steps}" ]]; then
    train_args+=(--lr-scheduler.total-steps "${lr_total_steps}")
fi
if [[ -n "${projector_kind}" ]]; then
    train_args+=(--projector-kind "${projector_kind}")
fi
if [[ -n "${projector_norm}" ]]; then
    train_args+=(--projector-norm "${projector_norm}")
fi
if [[ -n "${projector_ffn}" ]]; then
    train_args+=(--projector-ffn "${projector_ffn}")
fi
train_args+=(--projector-visual-layer-indices "${projector_visual_layers[@]}")
train_args+=(--projector-language-layer-indices "${projector_language_layers[@]}")
train_args+=(--projector-num-query-heads "${projector_num_query_heads}")
train_args+=(--projector-num-key-value-heads "${projector_num_key_value_heads}")
if [[ "${tie_projector_qkvo}" == "1" ]]; then
    train_args+=(--tie-projector-qkvo)
else
    train_args+=(--no-tie-projector-qkvo)
fi
if [[ -n "${projector_extra_merge_size}" ]]; then
    train_args+=(--projector-extra-merge-size "${projector_extra_merge_size}")
fi
if [[ -n "${projector_q_bucket}" ]]; then
    train_args+=(--projector-q-bucket "${projector_q_bucket}")
fi
if [[ -n "${projector_kv_bucket}" ]]; then
    train_args+=(--projector-kv-bucket "${projector_kv_bucket}")
fi
if [[ "${tracking}" == "1" ]]; then
    # SwanLab mirrors W&B as a connectivity backup; enable them together.
    train_args+=(--metrics.enable-wandb)
    train_args+=(--metrics.enable-swanlab)
fi
if [[ "${nvml_metrics}" == "1" ]]; then
    train_args+=(--metrics.enable-nvml-metrics)
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
if [[ "${dataloader_persistent_workers}" == "1" ]]; then
    train_args+=(--dataloader.persistent-workers)
else
    train_args+=(--dataloader.no-persistent-workers)
fi
if [[ "${dataloader_pin_memory}" == "1" ]]; then
    train_args+=(--dataloader.pin-memory)
else
    train_args+=(--dataloader.no-pin-memory)
fi
train_args+=("${train_extra_args[@]}")

echo
echo "==> Step 3/4: Training"
train_env=(
    "PYTORCH_CUDA_ALLOC_CONF=${pytorch_cuda_alloc_conf}"
    "OMP_NUM_THREADS=${OMP_NUM_THREADS:-${omp_num_threads}}"
    "TORCH_NCCL_ASYNC_ERROR_HANDLING=${torch_nccl_async_error_handling}"
    "TORCH_NCCL_ENABLE_MONITORING=${torch_nccl_enable_monitoring}"
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${torch_nccl_heartbeat_timeout_sec}"
    "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=${torch_nccl_wait_timeout_dump_milsec}"
    "TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=${torch_nccl_log_cpp_stack_on_unclean_shutdown}"
    "TORCH_NCCL_ENABLE_TIMING=${torch_nccl_enable_timing}"
    "TORCH_NCCL_NAN_CHECK=${torch_nccl_nan_check}"
    "TORCH_FR_CPP_STACK=${torch_nccl_trace_cpp_stack}"
    "TORCH_NCCL_DESYNC_DEBUG=${torch_nccl_desync_debug}"
    "WANDB_RUN_NAME=${tracking_run_name}"
)
if [[ "${python_faulthandler}" == "1" ]]; then
    train_env+=("PYTHONFAULTHANDLER=1")
fi
if [[ "${torch_show_cpp_stacktraces}" == "1" ]]; then
    train_env+=("TORCH_SHOW_CPP_STACKTRACES=1")
fi
if [[ "${torch_disable_addr2line}" == "1" ]]; then
    train_env+=("TORCH_DISABLE_ADDR2LINE=1")
fi
if [[ -n "${torchinductor_cache_dir}" ]]; then
    train_env+=("TORCHINDUCTOR_CACHE_DIR=${torchinductor_cache_dir}")
fi
train_env+=("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0")
if [[ "${torchinductor_nan_asserts}" == "1" ]]; then
    train_env+=("TORCHINDUCTOR_NAN_ASSERTS=1")
fi
if [[ "${torchinductor_runtime_triton_nan_asserts}" == "1" ]]; then
    train_env+=("TORCHINDUCTOR_RUNTIME_TRITON_NAN_ASSERTS=1")
fi
if [[ "${triton_debug}" == "1" ]]; then
    train_env+=("TRITON_DEBUG=1")
fi
if [[ -n "${torch_logs}" ]]; then
    train_env+=("TORCH_LOGS=${torch_logs}")
fi
if [[ -n "${torch_cpp_log_level}" ]]; then
    train_env+=("TORCH_CPP_LOG_LEVEL=${torch_cpp_log_level}")
fi
if [[ -n "${torch_distributed_debug}" ]]; then
    train_env+=("TORCH_DISTRIBUTED_DEBUG=${torch_distributed_debug}")
fi
if [[ -n "${flex_attention_log_file}" ]]; then
    train_env+=("TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE=${flex_attention_log_file}")
fi
if [[ -n "${nccl_debug}" ]]; then
    train_env+=("NCCL_DEBUG=${nccl_debug}")
fi
if [[ -n "${nccl_debug_subsys}" ]]; then
    train_env+=("NCCL_DEBUG_SUBSYS=${nccl_debug_subsys}")
fi
if [[ -n "${nccl_debug_file}" ]]; then
    train_env+=("NCCL_DEBUG_FILE=${nccl_debug_file}")
fi
if [[ "${torch_nccl_flight_recorder}" == "1" ]]; then
    train_env+=(
        "TORCH_NCCL_DUMP_ON_TIMEOUT=1"
        "TORCH_FR_BUFFER_SIZE=${torch_nccl_trace_buffer_size}"
    )
else
    train_env+=(
        "TORCH_NCCL_DUMP_ON_TIMEOUT=0"
        "TORCH_FR_BUFFER_SIZE=0"
    )
fi
env "${train_env[@]}" "${torchrun_cmd}" \
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
    --export_dtype "${export_dtype}" \
    "${convert_proj_args[@]}"

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
if [[ -z "${resume_dcp_path}" ]]; then
    echo "  Initial DCP checkpoint: ${dcp_dir}"
else
    echo "  Resumed from DCP:       ${initial_load_dcp}"
fi
echo "  Training outputs:       ${train_dump_dir}"
echo "  Final HF checkpoint:    ${final_hf_dir}"
