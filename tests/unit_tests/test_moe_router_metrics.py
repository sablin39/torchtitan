# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch
import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.trainer import Trainer


class TestTokenChoiceTopKRouterMetrics(unittest.TestCase):
    def test_consume_router_metrics_reports_max_vio_and_resets(self):
        router = TokenChoiceTopKRouter.Config(
            num_experts=4,
            top_k=2,
            score_func="softmax",
            gate=Linear.Config(in_features=4, out_features=4, bias=False),
        ).build()
        with torch.no_grad():
            router.gate.weight.copy_(torch.eye(4))

        x = torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 1.0, 0.0],
                [0.0, 4.0, 3.0, 1.0],
                [0.0, 1.0, 4.0, 3.0],
            ]
        )

        _, topk_expert_ids, scores = router(x)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids,
            True,
        )
        num_tokens_per_expert = routing_map.sum(dim=0)
        router._record_router_metrics(
            scores=scores,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        torch.testing.assert_close(num_tokens_per_expert, torch.tensor([2, 3, 2, 1]))
        metrics = router.consume_router_metrics()

        self.assertIn("token_entropy", metrics)
        self.assertIn("load_entropy", metrics)
        self.assertAlmostEqual(metrics["MaxVio"], 0.5, places=6)
        self.assertEqual(router.consume_router_metrics(), {})

    def test_trainer_collects_metrics_through_router_interface(self):
        class FakeMoE(nn.Module):
            def consume_router_metrics(self):
                return {"MaxVio": 0.25, "custom_router_metric": 2.0}

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_enabled = True
                self.moe = FakeMoE()

        class FakeModelPart(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict({"3": FakeBlock()})

        trainer = types.SimpleNamespace(model_parts=[FakeModelPart()])

        self.assertEqual(
            Trainer._collect_moe_metrics(trainer),
            {
                "moe/MaxVio": {"3": 0.25},
                "moe/custom_router_metric": {"3": 2.0},
            },
        )

    def test_trainer_collects_metrics_from_llm_layers(self):
        class FakeMoE(nn.Module):
            def consume_router_metrics(self):
                return {"MaxVio": 0.5}

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_enabled = True
                self.moe = FakeMoE()

        class FakeModelPart(nn.Module):
            def __init__(self):
                super().__init__()
                self.llm = nn.Module()
                self.llm.layers = nn.ModuleDict({"1": FakeBlock()})

        trainer = types.SimpleNamespace(model_parts=[FakeModelPart()])

        self.assertEqual(
            Trainer._collect_moe_metrics(trainer),
            {"moe/MaxVio": {"1": 0.5}},
        )

    def test_load_balancing_hook_registers_for_llm_layers(self):
        class FakeOptimizers:
            def __init__(self):
                self.hooks = []

            def register_step_pre_hook(self, hook):
                self.hooks.append(hook)

        class FakeMoE(nn.Module):
            load_balance_coeff = 1e-3

            def __init__(self):
                super().__init__()
                self.register_buffer("tokens_per_expert", torch.tensor([3.0, 1.0]))
                self.register_buffer("expert_bias", torch.zeros(2))

        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_enabled = True
                self.moe = FakeMoE()

        class FakeModelPart(nn.Module):
            def __init__(self):
                super().__init__()
                self.llm = nn.Module()
                self.llm.layers = nn.ModuleDict({"1": FakeBlock()})

        class FakeParallelDims:
            ep_enabled = False
            tp = 1

            def get_optional_mesh(self, name):
                del name
                return None

        optimizers = FakeOptimizers()
        model_part = FakeModelPart()
        register_moe_load_balancing_hook(
            optimizers,
            [model_part],
            FakeParallelDims(),
        )

        self.assertEqual(len(optimizers.hooks), 1)
        optimizers.hooks[0]()
        moe = model_part.llm.layers["1"].moe
        self.assertEqual(moe.tokens_per_expert.tolist(), [0.0, 0.0])
        self.assertFalse(torch.equal(moe.expert_bias, torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
