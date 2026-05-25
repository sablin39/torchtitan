# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


def _apply_rwkv_vl_projector_overrides(model_config, args):
    proj_overrides = {}
    for src_attr, dst_attr in (
        ("projector_kind", "kind"),
        ("projector_norm", "norm"),
        ("projector_ffn", "ffn"),
        ("projector_num_heads", "num_heads"),
        ("projector_head_dim", "head_dim"),
        ("projector_extra_merge_size", "extra_merge_size"),
    ):
        value = getattr(args, src_attr, None)
        if value is not None:
            proj_overrides[dst_attr] = value
    if proj_overrides and hasattr(model_config, "proj"):
        model_config.proj = replace(model_config.proj, **proj_overrides)
    if hasattr(model_config, "processor_spatial_merge_size") and hasattr(
        model_config, "vision_encoder"
    ):
        vision_merge = model_config.vision_encoder.spatial_merge_size
        extra = model_config.proj.extra_merge_size
        model_config.processor_spatial_merge_size = vision_merge * extra
    return model_config


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
    args=None,
):
    # load model and model args so that we can get the state dict shape
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model
    if args is not None:
        _apply_rwkv_vl_projector_overrides(model_config, args)

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    parser.add_argument("--projector_kind", type=str, default=None)
    parser.add_argument("--projector_norm", type=str, default=None)
    parser.add_argument("--projector_ffn", type=str, default=None)
    parser.add_argument("--projector_num_heads", type=int, default=None)
    parser.add_argument("--projector_head_dim", type=int, default=None)
    parser.add_argument("--projector_extra_merge_size", type=int, default=None)
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        args=args,
    )
