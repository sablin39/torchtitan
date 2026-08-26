# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.trainer import Trainer

from . import model_registry


def rwkv7_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(l2_wrap_factor=1e-4)
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(),
        model_spec=model_registry("debugmodel"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        # bf16 is mandatory: the FLA delta-rule chunk kernels do not support
        # fp32 (the bwd kernel exceeds shared-memory limits on most GPUs and
        # FLA itself emits a "ChunkDeltaRuleFunction does not support float32"
        # warning).
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * 512,
            max_context_length=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=SelectiveAC.Config(),
    )


# Pair RWKV7 LM training with the local Qwen3-VL tokenizer assets. Its 151669
# token IDs are padded to 151680 for checkpoint-compatible embedding shapes.
_QWEN3_VL_TOKENIZER_PATH = "./assets/hf/Qwen3-VL-2B-Instruct"
_QWEN3_VL_PADDED_VOCAB_SIZE = 151680


def _rwkv7_lm_config(
    flavor: str,
    *,
    lr: float = 8e-4,
    local_batch_size: int = 1,
    seq_len: int = 4096,
    steps: int = 10000,
    warmup_steps: int = 200,
    dataset: str = "fineweb-edu",
    hf_assets_path: str = _QWEN3_VL_TOKENIZER_PATH,
) -> Trainer.Config:
    """Trainer config for RWKV7 text-LM pretraining on fineweb-edu.

    Optimizer hyperparameters follow the RWKV-LM v7 pile recipe:
    AdamW(beta=(0.9, 0.99), eps=1e-18, wd=0.1) with the L2Wrap CE penalty at
    factor 1e-4. The tokenizer at ``hf_assets_path`` drives the model's
    fixed padded vocabulary size. Defaults assume single-node FSDP/bf16;
    override ``--training.num-tokens-per-microbatch-per-dp-rank``,
    ``--training.max-context-length``, and ``--training.steps`` per cluster.
    """
    model_spec = model_registry(flavor)
    model_spec.model.vocab_size = _QWEN3_VL_PADDED_VOCAB_SIZE
    model_spec.model.llm.vocab_size = _QWEN3_VL_PADDED_VOCAB_SIZE
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(l2_wrap_factor=1e-4)
        ),
        hf_assets_path=hf_assets_path,
        tokenizer=HuggingFaceTokenizer.Config(),
        model_spec=model_spec,
        optimizer=default_adamw(lr=lr, betas=(0.9, 0.99), eps=1e-18, weight_decay=0.1),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps, decay_type="cosine", min_lr_factor=0.0375
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
            max_context_length=seq_len,
            steps=steps,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS[dataset])
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(interval=1000, last_save_model_only=True),
        # Full activation checkpointing is the standard for >1B-param
        # pretraining: recompute every layer's activations in backward so
        # peak memory tracks params + opt + grads, not activations.
        activation_checkpoint=FullAC.Config(),
    )


def rwkv7_0_4b() -> Trainer.Config:
    return _rwkv7_lm_config("0.4B")


def rwkv7_1_5b() -> Trainer.Config:
    return _rwkv7_lm_config("1.5B", lr=6e-4)


def rwkv7_2_9b() -> Trainer.Config:
    return _rwkv7_lm_config("2.9B", lr=4e-4)


def rwkv7_moe_3b() -> Trainer.Config:
    return _rwkv7_lm_config(
        "3B-MoE",
        lr=3e-4,
        local_batch_size=2,
        warmup_steps=2000,
    )


def rwkv7_7_2b() -> Trainer.Config:
    return _rwkv7_lm_config("7.2B", lr=3e-4)


def rwkv7_13_3b() -> Trainer.Config:
    return _rwkv7_lm_config("13.3B", lr=2e-4)
