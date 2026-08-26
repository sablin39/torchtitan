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
from torch.distributed.checkpoint import HuggingFaceStorageReader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchtitan.components.checkpointer import ModelWrapper


def _apply_rwkv_vl_projector_overrides(model_config, args):
    """Patch the rwkv_vl model config in-place from CLI projector overrides.

    Used by the HF↔DCP conversion scripts so that the torchtitan-side model
    construction matches the projector variant baked into the HF checkpoint.
    The processor merge size is derived from ``projector_extra_merge_size``
    and the vision encoder's ``spatial_merge_size`` so the two stay in sync.
    """
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
def convert_from_hf(input_dir, output_dir, model_name, model_flavor, args=None):
    # initialize model to allocate memory for state dict
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model
    if args is not None:
        _apply_rwkv_vl_projector_overrides(model_config, args)

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    sd_adapter = model_spec.state_dict_adapter(model_config, None)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from HF to DCP safetensors format, but sd_adapter is not provided."
    # get state dict in tt format with allocated memory
    state_dict = model._get_state_dict()
    # convert empty state dict to hf format so that hf weights can be loaded into it
    hf_state_dict = sd_adapter.to_hf(state_dict)
    dcp.load(
        hf_state_dict,
        storage_reader=HuggingFaceStorageReader(path=input_dir),
    )
    # convert state dict format back hf->tt and save
    state_dict = sd_adapter.from_hf(hf_state_dict)
    dcp.save(
        state_dict,
        checkpoint_id=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with HF checkpoint"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    # rwkv_vl projector overrides (ignored by other model families).
    parser.add_argument("--projector_kind", type=str, default=None)
    parser.add_argument("--projector_norm", type=str, default=None)
    parser.add_argument("--projector_ffn", type=str, default=None)
    parser.add_argument("--projector_num_heads", type=int, default=None)
    parser.add_argument("--projector_head_dim", type=int, default=None)
    parser.add_argument("--projector_extra_merge_size", type=int, default=None)
    args = parser.parse_args()

    convert_from_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args=args,
    )
