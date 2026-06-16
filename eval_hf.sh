#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python_cmd="${PYTHON:-${repo_root}/.venv/bin/python}"
lmms_eval_root="${LMMS_EVAL_ROOT:-${repo_root}/../lmms-eval}"
ckpt="${CKPT:-/mnt/raid0_8t/rwkv_vl_0.4b/hf_step_79803}"
tasks="${TASKS:-chartqa}"
batch_size="${BATCH_SIZE:-1}"
limit="${LIMIT:-64}"
output_path="${OUTPUT_PATH:-${repo_root}/eval_output/rwkv_vl_0.4b_step_79803}"
gen_kwargs="${GEN_KWARGS:-max_new_tokens=16,temperature=0,do_sample=false}"
model_args="${MODEL_ARGS:-pretrained=${ckpt},trust_remote_code=True,use_cache=false}"
reasoning_tags="${REASONING_TAGS:-[[\"<think>\",\"</think>\"],[\"<analysis>\",\"</analysis>\"]]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -d "${lmms_eval_root}/lmms_eval" ]]; then
    export PYTHONPATH="${ckpt}:${repo_root}:${lmms_eval_root}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${ckpt}:${repo_root}:${PYTHONPATH:-}"
fi

args=(
    --model huggingface
    --model_args "${model_args}"
    --tasks "${tasks}"
    --batch_size "${batch_size}"
    --output_path "${output_path}"
    --gen_kwargs "${gen_kwargs}"
    --reasoning_tags "${reasoning_tags}"
    --process_with_media
)

if [[ -n "${limit}" ]]; then
    args+=(--limit "${limit}")
fi
if [[ "${LOG_SAMPLES:-1}" == "1" ]]; then
    args+=(--log_samples)
fi

echo "Checkpoint: ${ckpt}"
echo "Tasks:      ${tasks}"
echo "Limit:      ${limit:-full}"
echo "Output:     ${output_path}"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "lmms-eval:  ${lmms_eval_root}"

"${python_cmd}" -m lmms_eval eval "${args[@]}"
