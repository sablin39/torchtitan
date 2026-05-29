# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest

import torch
import torch.nn as nn

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

        _, _, num_tokens_per_expert, _ = router(x)

        torch.testing.assert_close(
            num_tokens_per_expert, torch.tensor([2.0, 3.0, 2.0, 1.0])
        )
        metrics = router.consume_router_metrics()

        self.assertIn("token_entropy", metrics)
        self.assertIn("load_entropy", metrics)
        self.assertAlmostEqual(metrics["MaxVio"], 0.5, places=6)
        self.assertEqual(router.consume_router_metrics(), {})

    def test_trainer_collects_metrics_through_router_interface(self):
        class FakeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe_enabled = True
                self.moe = FakeMoE()

        class FakeMoE(nn.Module):
            def consume_router_metrics(self):
                return {"MaxVio": 0.25, "custom_router_metric": 2.0}

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


if __name__ == "__main__":
    unittest.main()
