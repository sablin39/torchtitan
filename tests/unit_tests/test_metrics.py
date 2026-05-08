# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import tempfile
import types
import unittest
from unittest import mock

from torchtitan.components.metrics import WandBLogger
from torchtitan.config import ConfigManager


class MetricsTest(unittest.TestCase):
    def test_swanlab_sync_happens_before_wandb_init(self):
        call_order = []

        fake_swanlab = types.SimpleNamespace(
            sync_wandb=mock.Mock(
                side_effect=lambda **_kwargs: call_order.append("swanlab.sync")
            )
        )
        fake_wandb = types.SimpleNamespace(
            init=mock.Mock(side_effect=lambda **_kwargs: call_order.append("wandb.init")),
            run=object(),
            finish=mock.Mock(),
        )

        with (
            mock.patch.dict(
                sys.modules,
                {"swanlab": fake_swanlab, "wandb": fake_wandb},
            ),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            WandBLogger(tmpdir, sync_swanlab=True)

        self.assertEqual(call_order, ["swanlab.sync", "wandb.init"])
        fake_swanlab.sync_wandb.assert_called_once_with(wandb_run=True)
        fake_wandb.init.assert_called_once()

    def test_swanlab_only_disables_wandb_run_before_wandb_init(self):
        call_order = []

        fake_swanlab = types.SimpleNamespace(
            sync_wandb=mock.Mock(
                side_effect=lambda **_kwargs: call_order.append("swanlab.sync")
            )
        )
        fake_wandb = types.SimpleNamespace(
            init=mock.Mock(side_effect=lambda **_kwargs: call_order.append("wandb.init")),
            run=object(),
            finish=mock.Mock(),
        )

        with (
            mock.patch.dict(
                sys.modules,
                {"swanlab": fake_swanlab, "wandb": fake_wandb},
            ),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            WandBLogger(tmpdir, sync_swanlab=True, swanlab_wandb_run=False)

        self.assertEqual(call_order, ["swanlab.sync", "wandb.init"])
        fake_swanlab.sync_wandb.assert_called_once_with(wandb_run=False)
        fake_wandb.init.assert_called_once()

    def test_cli_override_enable_swanlab(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--metrics.enable-swanlab",
            ]
        )
        assert config.metrics.enable_swanlab


if __name__ == "__main__":
    unittest.main()
