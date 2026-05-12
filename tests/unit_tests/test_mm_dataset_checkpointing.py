# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.hf_datasets.multimodal.mm_datasets import MMDataLoader


_TOKENIZER_PATH = "tests/assets/tokenizer"

_TOKENIZER_CONFIG = MultiModalTokenizer.Config(
    image_token="<|image_pad|>",
    video_token="<|video_pad|>",
    vision_start_token="<|vision_start|>",
    vision_end_token="<|vision_end|>",
    pad_token="<|endoftext|>",
)


_TOKENIZER = _TOKENIZER_CONFIG.build(tokenizer_path=_TOKENIZER_PATH)


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, key) for item in value)
    return False


class TestMMDatasetCheckpointing(unittest.TestCase):
    """Test save/load for multimodal dataset, mirroring test_dataset_checkpointing.py."""

    def _build_dataloader(self, batch_size, seq_len, world_size, rank):
        dl_config = MMDataLoader.Config(
            dataset="cc12m-test",
            max_images_per_batch=128,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=784,
            max_pixels=200000,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )

        return dl_config.build(
            dp_world_size=world_size,
            dp_rank=rank,
            tokenizer=_TOKENIZER,
            seq_len=seq_len,
            local_batch_size=batch_size,
        )

    def test_cc12m_resumption(self):
        for world_size in [1, 2]:
            for rank in range(world_size):
                batch_size = 1
                seq_len = 4096

                dl = self._build_dataloader(batch_size, seq_len, world_size, rank)

                it = iter(dl)
                for _ in range(5):
                    next(it)
                state = dl.state_dict()

                # Create new dataloader, restore checkpoint, verify subsequent
                # batches match
                dl_resumed = self._build_dataloader(
                    batch_size, seq_len, world_size, rank
                )
                dl_resumed.load_state_dict(state)
                it_resumed = iter(dl_resumed)

                for _ in range(10):
                    expected_input, expected_labels = next(it)
                    input_dict, labels = next(it_resumed)
                    assert torch.equal(
                        input_dict["input"], expected_input["input"]
                    ), f"input_ids mismatch (world_size={world_size}, rank={rank})"
                    assert torch.equal(
                        labels, expected_labels
                    ), f"labels mismatch (world_size={world_size}, rank={rank})"
                    assert torch.equal(
                        input_dict["positions"], expected_input["positions"]
                    ), f"positions mismatch (world_size={world_size}, rank={rank})"
                    for key in ["pixel_values", "grid_thw"]:
                        exp_v = expected_input[key]
                        res_v = input_dict[key]
                        assert (exp_v is None) == (
                            res_v is None
                        ), f"{key} None mismatch (world_size={world_size}, rank={rank})"
                        if exp_v is not None:
                            assert exp_v.shape == res_v.shape, (
                                f"{key} shape mismatch: {exp_v.shape} vs {res_v.shape} "
                                f"(world_size={world_size}, rank={rank})"
                            )

    def test_checkpoint_state_drops_processed_packer_samples(self):
        dl_config = MMDataLoader.Config(
            dataset="cc12m-test",
            packing_buffer_size=64,
            max_images_per_batch=128,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=784,
            max_pixels=200000,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
        dataloader = dl_config.build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=_TOKENIZER,
            seq_len=4096,
            local_batch_size=1,
        )
        dataset = dataloader.dataset
        sample = next(iter(dataset._data))
        processed = dataset.sample_processor(
            sample=sample,
            tokenizer=dataset._tokenizer,
            patch_size=dataset.patch_size,
            temporal_patch_size=dataset.temporal_patch_size,
            spatial_merge_size=dataset.spatial_merge_size,
            min_pixels=dataset.min_pixels,
            max_pixels=dataset.max_pixels,
            image_mean=dataset.image_mean,
            image_std=dataset.image_std,
            video_dir=dataset.video_dir,
            video_fps=dataset.video_fps,
            video_min_frames=dataset.video_min_frames,
            video_max_frames=dataset.video_max_frames,
            seq_len=dataset.seq_len,
        )
        self.assertIsNotNone(processed)
        self.assertIn("pixel_values", processed)

        dataset.packer.add_sample(processed)
        self.assertEqual(len(dataset.packer._sample_buffer), 1)
        state = dataset.state_dict()
        self.assertNotIn("packer_state", state)
        self.assertFalse(_contains_key(state, "pixel_values"))

        dataset.load_state_dict(state)
        self.assertEqual(dataset.packer._sample_buffer, {})
        self.assertEqual(dataset.packer._next_id, 0)
        self.assertEqual(len(dataset.packer.packed_samples), 0)


if __name__ == "__main__":
    unittest.main()
