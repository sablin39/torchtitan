# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import sys
import types
import unittest
from unittest import mock

from torchtitan.tools.nvml_metrics import NvmlGpuMetricsMonitor


class _FakeMetric:
    def __init__(self):
        self.metricId = 0
        self.nvmlReturn = 0
        self.value = 0.0


class _FakeGpmRequest:
    def __init__(self):
        self.version = 0
        self.numMetrics = 0
        self.sample1 = None
        self.sample2 = None
        self.metrics = [_FakeMetric() for _ in range(128)]


class NvmlMetricsTest(unittest.TestCase):
    def test_missing_pynvml_raises(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ModuleNotFoundError("No module named 'pynvml'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ModuleNotFoundError):
                NvmlGpuMetricsMonitor(device_index=0, warn=False)

    def test_returns_gpm_metrics_as_logger_ready_dict(self):
        fake_pynvml = types.SimpleNamespace()
        fake_pynvml.NVML_SUCCESS = 0
        fake_pynvml.NVML_GPM_METRICS_GET_VERSION = 1
        fake_pynvml.NVML_GPM_METRIC_SM_UTIL = 2
        fake_pynvml.NVML_GPM_METRIC_ANY_TENSOR_UTIL = 5
        fake_pynvml.c_nvmlGpmMetricsGet_t = _FakeGpmRequest
        fake_pynvml.nvmlInit = mock.Mock()
        fake_pynvml.nvmlShutdown = mock.Mock()
        fake_pynvml.nvmlDeviceGetHandleByIndex = mock.Mock(return_value="handle")
        fake_pynvml.nvmlGpmQueryDeviceSupport = mock.Mock(
            return_value=types.SimpleNamespace(isSupportedDevice=1)
        )
        fake_pynvml.nvmlGpmSampleAlloc = mock.Mock(side_effect=["sample1", "sample2"])
        fake_pynvml.nvmlGpmSampleGet = mock.Mock(side_effect=lambda *_args: None)
        fake_pynvml.nvmlGpmSampleFree = mock.Mock()

        def nvml_gpm_metrics_get(request):
            for i in range(request.numMetrics):
                request.metrics[i].nvmlReturn = fake_pynvml.NVML_SUCCESS
                request.metrics[i].value = 10.0 + i
            return request

        fake_pynvml.nvmlGpmMetricsGet = mock.Mock(side_effect=nvml_gpm_metrics_get)

        with mock.patch.dict(sys.modules, {"pynvml": fake_pynvml}):
            monitor = NvmlGpuMetricsMonitor(
                device_index=0,
                metrics=(
                    # Keep the test focused: no dependence on the full default list.
                    types.SimpleNamespace(
                        name="gpu/sm_util(%)",
                        metric_id_name="NVML_GPM_METRIC_SM_UTIL",
                    ),
                    types.SimpleNamespace(
                        name="gpu/tensor_util(%)",
                        metric_id_name="NVML_GPM_METRIC_ANY_TENSOR_UTIL",
                    ),
                ),
                warn=False,
            )
            self.assertEqual(
                monitor.get_metrics(),
                {
                    "gpu/sm_util(%)": 10.0,
                    "gpu/tensor_util(%)": 11.0,
                },
            )
            monitor.close()

    def test_unsupported_gpm_returns_empty_dict_without_coarse_utilization(self):
        fake_pynvml = types.SimpleNamespace()
        fake_pynvml.nvmlInit = mock.Mock()
        fake_pynvml.nvmlShutdown = mock.Mock()
        fake_pynvml.nvmlDeviceGetHandleByIndex = mock.Mock(return_value="handle")
        fake_pynvml.nvmlGpmQueryDeviceSupport = mock.Mock(
            return_value=types.SimpleNamespace(isSupportedDevice=0)
        )
        fake_pynvml.nvmlDeviceGetUtilizationRates = mock.Mock(
            side_effect=AssertionError("coarse utilization should not be called")
        )

        with mock.patch.dict(sys.modules, {"pynvml": fake_pynvml}):
            monitor = NvmlGpuMetricsMonitor(device_index=0, warn=False)
            self.assertEqual(monitor.get_metrics(), {})
            fake_pynvml.nvmlDeviceGetUtilizationRates.assert_not_called()
            monitor.close()


if __name__ == "__main__":
    unittest.main()
