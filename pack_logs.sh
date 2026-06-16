#!/usr/bin/env bash
# Pack the lightweight logs needed to debug a failed run_vl_train.sh training.

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage: ./pack_logs.sh RUN_DIR [ARCHIVE]

Pack logs from a run_vl_train.sh artifact directory.

Included when present:
  train/
  terminal.log
  flex_attention_autotune.json

RUN_DIR should usually be an outputs/rwkv_vl_train_<timestamp> directory.
ARCHIVE defaults to ./<run-name>_logs_<timestamp>.tar.xz.
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

run_dir="${1%/}"
if [[ ! -d "${run_dir}" ]]; then
    echo "RUN_DIR does not exist or is not a directory: ${run_dir}" >&2
    exit 1
fi

run_name="$(basename -- "${run_dir}")"
pack_timestamp="$(date +%Y%m%d_%H%M%S)"
archive="${2:-${run_name}_logs_${pack_timestamp}.tar.xz}"
if [[ "${archive}" != /* ]]; then
    archive="${PWD}/${archive}"
fi

items=()
missing=()

add_if_exists() {
    local rel_path="$1"
    if [[ -e "${run_dir}/${rel_path}" ]]; then
        items+=("${rel_path}")
    else
        missing+=("${rel_path}")
    fi
}

add_if_exists "train"

if [[ -e "${run_dir}/terminal.log" ]]; then
    items+=("terminal.log")
elif [[ -e "${run_dir}/train/terminal.log" ]]; then
    # run_vl_train.sh writes terminal.log here when terminal_log_file="auto".
    :
else
    missing+=("terminal.log or train/terminal.log")
fi

add_if_exists "flex_attention_autotune.json"

if [[ ${#items[@]} -eq 0 ]]; then
    echo "No log artifacts found under: ${run_dir}" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${archive}")"
tar -C "${run_dir}" -cJf "${archive}" "${items[@]}"

echo "Packed: ${archive}"
echo "Included:"
printf '  %s\n' "${items[@]}"

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing:"
    printf '  %s\n' "${missing[@]}"
fi
