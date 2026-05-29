# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry


def rwkv7_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(l2_wrap_factor=1e-4),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=HuggingFaceTokenizer.Config(),
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        # bf16 is mandatory: the FLA delta-rule chunk kernels do not support
        # fp32 (the bwd kernel exceeds shared-memory limits on most GPUs and
        # FLA itself emits a "ChunkDeltaRuleFunction does not support float32"
        # warning).
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=512,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(interval=10, last_save_model_only=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
    )


# Pair RWKV7 LM training with the Qwen3-VL HF tokenizer assets. The model's
# embedding / lm_head vocab is snapped to this tokenizer automatically at
# trainer-build time (see ``align_vocab_size_to_tokenizer``).
_QWEN3_VL_TOKENIZER_PATH = "./assets/hf/Qwen3-VL-2B-Instruct"


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
    vocab_size (via the ``update_from_config(tokenizer=...)`` hook), so
    swapping tokenizers does not require changes here. Defaults assume
    single-node FSDP/bf16; override ``--training.local_batch_size`` and
    ``--training.steps`` per cluster.
    """
    return Trainer.Config(
        loss=ChunkedCELoss.Config(l2_wrap_factor=1e-4),
        hf_assets_path=hf_assets_path,
        tokenizer=HuggingFaceTokenizer.Config(),
        model_spec=model_registry(flavor),
        optimizer=OptimizersContainer.Config(
            lr=lr, beta1=0.9, beta2=0.99, eps=1e-18, weight_decay=0.1
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps, decay_type="cosine", min_lr_factor=0.0375
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=steps,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset=dataset),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(interval=1000, last_save_model_only=True),
        # Full activation checkpointing is the standard for >1B-param
        # pretraining: recompute every layer's activations in backward so
        # peak memory tracks params + opt + grads, not activations.
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def rwkv7_0_4b() -> Trainer.Config:
    return _rwkv7_lm_config("0.4B")


def rwkv7_1_5b() -> Trainer.Config:
    return _rwkv7_lm_config("1.5B", lr=6e-4)


def rwkv7_2_9b() -> Trainer.Config:
    return _rwkv7_lm_config("2.9B", lr=4e-4)


def rwkv7_moe_3b() -> Trainer.Config:
    return Trainer.Config(
        loss=ChunkedCELoss.Config(l2_wrap_factor=1e-4),
        hf_assets_path=_QWEN3_VL_TOKENIZER_PATH,
        tokenizer=HuggingFaceTokenizer.Config(),
        model_spec=model_registry("3B-MoE"),
        optimizer=OptimizersContainer.Config(
            name="AdamW", lr=3e-4, beta1=0.9, beta2=0.99, eps=1e-18, weight_decay=0.1
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000, decay_type="linear", min_lr_factor=1.0 # WSM strategy
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=1e10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="fineweb-edu",
            dataset_path="/mnt/raid0_8t/fineweb-edu/sample/10BT/",
            infinite=False
        ),
        parallelism=ParallelismConfig(
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            context_parallel_degree=1,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(interval=1000, 
                                            last_save_model_only=False,
                                            export_dtype="bfloat16",
                                            keep_latest_k=0),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
    )


def rwkv7_7_2b() -> Trainer.Config:
    return _rwkv7_lm_config("7.2B", lr=3e-4)


def rwkv7_13_3b() -> Trainer.Config:
    return _rwkv7_lm_config("13.3B", lr=2e-4)
