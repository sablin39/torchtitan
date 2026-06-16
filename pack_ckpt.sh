#!/usr/bin/env bash
# Pack artifacts produced by run_vl_train.sh.

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage: ./pack_ckpt.sh --dir RUN_DIR [--logs | --latest]
       ./pack_ckpt.sh RUN_DIR [--logs | --latest]

RUN_DIR is the output_root directory produced by run_vl_train.sh.
Archives are written beside this script:

  <run-name>_log_only.tar.xz
  <run-name>_step_<N>.tar.xz
  <run-name>_step_<N>_all.tar.xz

Options:
  --dir DIR     run_vl_train.sh output_root directory.
  --logs        Pack only training logs.
  --latest      Pack training logs plus the latest DCP checkpoint.
                Without --latest, all DCP checkpoints are packed.
                Checkpoint archives also include hf_export when present.
  -h, --help    Show this help.

Compression is hard-coded to:
  xz -9e -T64 --memlimit-compress=256GiB
USAGE
}

die() {
    echo "$*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
run_dir=""
logs_only=0
latest_only=0
xz_args=(-9e -T64 --memlimit-compress=256GiB)
archive_tmp=""
checksum_tmp=""

cleanup() {
    rm -f "${list:-}"
    [[ -z "${archive_tmp}" ]] || rm -f "${archive_tmp}"
    [[ -z "${checksum_tmp}" ]] || rm -f "${checksum_tmp}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            [[ $# -ge 2 ]] || die "--dir requires a directory"
            run_dir="$2"
            shift 2
            ;;
        --dir=*)
            run_dir="${1#--dir=}"
            shift
            ;;
        --logs)
            logs_only=1
            shift
            ;;
        --latest)
            latest_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            usage
            die "Unknown option: $1"
            ;;
        *)
            [[ -z "${run_dir}" ]] || die "Unexpected argument: $1"
            run_dir="$1"
            shift
            ;;
    esac
done

[[ -n "${run_dir}" ]] || { usage; exit 2; }
[[ "${logs_only}" -eq 0 || "${latest_only}" -eq 0 ]] || die "--logs and --latest cannot be used together"

run_dir="${run_dir%/}"
[[ -d "${run_dir}" ]] || die "RUN_DIR is not a directory: ${run_dir}"

root="$(cd "${run_dir}" && pwd -P)"
parent="$(dirname -- "${root}")"
run_name="$(basename -- "${root}")"
train_dir="${root}/train"
checkpoint_dir="${train_dir}/checkpoint"
hf_export_dir="${root}/hf_export"

[[ -d "${train_dir}" ]] || die "Missing training output directory: ${train_dir}"

to_rel() {
    local path="$1"
    printf '%s\n' "${path#"${parent}/"}"
}

to_gb() {
    local bytes="$1"
    awk -v bytes="${bytes}" 'BEGIN { printf "%.2f", bytes / 1000000000 }'
}

list="$(mktemp)"
trap cleanup EXIT
items=()

add_path() {
    local path="$1"
    local rel
    rel="$(to_rel "${path}")"
    printf '%s\0' "${rel}" >> "${list}"
    items+=("${rel}")
}

add_training_logs() {
    local path
    while IFS= read -r -d '' path; do
        add_path "${path}"
    done < <(
        find "${train_dir}" \
            -path "${checkpoint_dir}" -prune \
            -o \( -type f -o -type l \) -print0
    )

    for path in "${root}/flex_attention_autotune" "${root}/flex_attention_autotune.json"; do
        [[ -e "${path}" ]] && add_path "${path}"
    done

    return 0
}

add_hf_export() {
    if [[ -e "${hf_export_dir}" ]]; then
        [[ -d "${hf_export_dir}" ]] || die "hf_export exists but is not a directory: ${hf_export_dir}"
        add_path "${hf_export_dir}"
    fi
}

mapfile -t step_dirs < <(
    if [[ -d "${checkpoint_dir}" ]]; then
        find "${checkpoint_dir}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' | sort -V
    fi
)

latest_step=""
latest_step_dir=""
if [[ "${#step_dirs[@]}" -gt 0 ]]; then
    latest_step_dir="${step_dirs[$((${#step_dirs[@]} - 1))]}"
    latest_step="$(basename -- "${latest_step_dir}")"
    latest_step="${latest_step#step-}"
fi

add_training_logs

archive_tag="log_only"
if [[ "${logs_only}" -eq 0 ]]; then
    [[ -n "${latest_step_dir}" ]] || die "No step-* checkpoints found under ${checkpoint_dir}"
    add_hf_export
    if [[ "${latest_only}" -eq 1 ]]; then
        add_path "${latest_step_dir}"
        archive_tag="step_${latest_step}"
    else
        for step_dir in "${step_dirs[@]}"; do
            add_path "${step_dir}"
        done
        archive_tag="step_${latest_step}_all"
    fi
fi

[[ "${#items[@]}" -gt 0 ]] || die "No files selected under ${root}"

archive="${script_dir}/${run_name}_${archive_tag}.tar.xz"
checksum="${archive}.sha256"
archive_tmp="${archive}.tmp.$$"
checksum_tmp="${checksum}.tmp.$$"
archive_name="$(basename -- "${archive}")"
checksum_name="$(basename -- "${checksum}")"
checksum_tmp_name="$(basename -- "${checksum_tmp}")"

size=0
while IFS= read -r -d '' rel; do
    bytes="$(du -sb -- "${parent}/${rel}" | awk '{print $1}')"
    size=$((size + bytes))
done < "${list}"
size_gb="$(to_gb "${size}")"

echo "Files/directories to archive:"
printf '  %s\n' "${items[@]}"
echo
echo "Estimated uncompressed size: ${size_gb} GB"
echo "Archive: ${archive}"
echo "Temporary archive: ${archive_tmp}"
echo "Compression: xz ${xz_args[*]}"
echo

if command -v pv >/dev/null 2>&1; then
    tar -C "${parent}" --null -cf - -T "${list}" \
        | pv -s "${size}" \
        | xz "${xz_args[@]}" \
        > "${archive_tmp}"
else
    tar -C "${parent}" --null -cf - -T "${list}" \
        | xz "${xz_args[@]}" \
        > "${archive_tmp}"
fi

mv -f "${archive_tmp}" "${archive}"
archive_tmp=""
archive_size="$(du -sb -- "${archive}" | awk '{print $1}')"
archive_size_gb="$(to_gb "${archive_size}")"
echo "Packed: ${archive} (${archive_size_gb} GB)"

(
    cd "${script_dir}"
    sha256sum "${archive_name}" > "${checksum_tmp_name}"
)
mv -f "${checksum_tmp}" "${checksum}"
checksum_tmp=""
echo "SHA256: ${checksum}"

remote_host="rwkv@47.115.88.183"
remote_port="2333"
remote_dir="/home/rwkv/molin/torchtitan"
remote_dir_q="$(printf "%q" "${remote_dir}")"
checksum_name_q="$(printf "%q" "${checksum_name}")"

rsync -ahP -e "ssh -p ${remote_port}" "${archive}" "${checksum}" "${remote_host}:${remote_dir}/"
ssh -p "${remote_port}" "${remote_host}" "cd ${remote_dir_q} && sha256sum -c ${checksum_name_q}"
