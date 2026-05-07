# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from torchtitan.tools.logging import logger, warn_once


@dataclass(frozen=True, slots=True)
class NvmlMetric:
    name: str
    metric_id_names: str | tuple[str, ...]


DEFAULT_GPM_METRICS = (
    NvmlMetric("gpu/sm_util(%)", "NVML_GPM_METRIC_SM_UTIL"),
    NvmlMetric("gpu/sm_occupancy(%)", "NVML_GPM_METRIC_SM_OCCUPANCY"),
    NvmlMetric("gpu/tensor_util(%)", "NVML_GPM_METRIC_ANY_TENSOR_UTIL"),
    NvmlMetric("gpu/tensor_dfma_util(%)", "NVML_GPM_METRIC_DFMA_TENSOR_UTIL"),
    NvmlMetric("gpu/tensor_hmma_util(%)", "NVML_GPM_METRIC_HMMA_TENSOR_UTIL"),
    NvmlMetric("gpu/tensor_dmma_util(%)", "NVML_GPM_METRIC_DMMA_TENSOR_UTIL"),
    NvmlMetric("gpu/tensor_imma_util(%)", "NVML_GPM_METRIC_IMMA_TENSOR_UTIL"),
    NvmlMetric("gpu/dram_bw_util(%)", "NVML_GPM_METRIC_DRAM_BW_UTIL"),
    NvmlMetric("gpu/fp64_util(%)", "NVML_GPM_METRIC_FP64_UTIL"),
    NvmlMetric("gpu/fp32_util(%)", "NVML_GPM_METRIC_FP32_UTIL"),
    NvmlMetric("gpu/fp16_util(%)", "NVML_GPM_METRIC_FP16_UTIL"),
    NvmlMetric("gpu/integer_util(%)", "NVML_GPM_METRIC_INTEGER_UTIL"),
    NvmlMetric("gpu/pcie_tx_mib_s", "NVML_GPM_METRIC_PCIE_TX_PER_SEC"),
    NvmlMetric("gpu/pcie_rx_mib_s", "NVML_GPM_METRIC_PCIE_RX_PER_SEC"),
    NvmlMetric("gpu/nvlink_tx_mib_s", "NVML_GPM_METRIC_NVLINK_TOTAL_TX_PER_SEC"),
    NvmlMetric("gpu/nvlink_rx_mib_s", "NVML_GPM_METRIC_NVLINK_TOTAL_RX_PER_SEC"),
)


class NvmlGpuMetricsMonitor:
    """Collect scalar NVIDIA GPU metrics that can be sent to W&B/TensorBoard.

    The monitor uses NVML's GPU Performance Monitoring (GPM) APIs, which expose
    SM, tensor pipe, DRAM, PCIe, and NVLink metrics on supported NVIDIA GPUs.

    ``get_metrics()`` returns a flat ``dict[str, float]`` so callers can merge it
    into TorchTitan's ``extra_metrics`` or pass it directly to loggers like W&B.
    """

    def __init__(
        self,
        *,
        device_index: int | None = None,
        enable_gpm: bool = True,
        prefix: str = "",
        metrics: tuple[NvmlMetric, ...] = DEFAULT_GPM_METRICS,
        warn: bool = True,
    ) -> None:
        self.device_index = (
            self._get_default_device_index() if device_index is None else device_index
        )
        self.enable_gpm = enable_gpm
        self.prefix = prefix
        self.metric_specs = metrics
        self.warn = warn

        self._pynvml: ModuleType | None = None
        self._handle: Any | None = None
        self._gpm_supported = False
        self._gpm_metric_ids: list[int] = []
        self._gpm_metric_names: list[str] = []
        self._previous_sample: Any | None = None
        self._initialized = False
        self._closed = False

        self._init_nvml()

    @property
    def is_available(self) -> bool:
        return self._initialized and self._handle is not None

    @property
    def gpm_supported(self) -> bool:
        return self._gpm_supported

    def get_metrics(self) -> dict[str, float]:
        """Return logger-ready metrics for the current local GPU.

        GPM metrics are computed over the interval between this call and the
        previous GPM sample. If GPM is unavailable, this returns an empty dict.
        """
        if self._closed or not self.is_available:
            return {}

        metrics: dict[str, float] = {}
        if self._gpm_supported:
            metrics.update(self._get_gpm_metrics())

        return metrics

    def close(self) -> None:
        if self._closed:
            return

        if self._pynvml is not None:
            if self._previous_sample is not None:
                try:
                    self._pynvml.nvmlGpmSampleFree(self._previous_sample)
                except Exception:
                    pass
                self._previous_sample = None
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

        self._closed = True

    def __enter__(self) -> NvmlGpuMetricsMonitor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _init_nvml(self) -> None:
        import pynvml

        self._pynvml = pynvml
        try:
            pynvml.nvmlInit()
            self._handle = self._get_device_handle(pynvml)
        except Exception as e:
            if self.warn:
                warn_once(logger, f"Failed to initialize NVML GPU metrics: {e}")
            return

        self._initialized = True
        if self.enable_gpm:
            self._init_gpm(pynvml)

    def _get_default_device_index(self) -> int:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.current_device()

    def _get_device_handle(self, pynvml: ModuleType) -> Any:
        uuid = self._get_torch_device_uuid()
        if uuid is not None:
            try:
                return pynvml.nvmlDeviceGetHandleByUUID(uuid)
            except Exception:
                pass

        pci_bus_id = self._get_torch_pci_bus_id()
        if pci_bus_id is not None:
            try:
                return pynvml.nvmlDeviceGetHandleByPciBusId(pci_bus_id)
            except Exception:
                pass

        return pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def _get_torch_device_uuid(self) -> str | bytes | None:
        if not torch.cuda.is_available():
            return None

        try:
            uuid = torch.cuda.get_device_properties(self.device_index).uuid
        except Exception:
            return None

        if isinstance(uuid, bytes):
            return uuid
        if isinstance(uuid, str):
            return uuid.encode()
        return None

    def _get_torch_pci_bus_id(self) -> str | bytes | None:
        if not torch.cuda.is_available():
            return None

        try:
            pci_bus_id = torch.cuda.get_device_properties(self.device_index).pci_bus_id
        except Exception:
            return None

        if isinstance(pci_bus_id, bytes):
            return pci_bus_id
        if isinstance(pci_bus_id, str):
            return pci_bus_id.encode()
        return None

    def _init_gpm(self, pynvml: ModuleType) -> None:
        try:
            support = pynvml.nvmlGpmQueryDeviceSupport(self._handle)
        except Exception as e:
            if self.warn:
                warn_once(logger, f"NVML GPM metrics are unavailable: {e}")
            return

        if not getattr(support, "isSupportedDevice", 0):
            if self.warn:
                warn_once(
                    logger,
                    "NVML GPM metrics are not supported on this GPU; tensor/SM "
                    "metrics are disabled.",
                )
            return

        metric_ids: list[int] = []
        metric_names: list[str] = []
        for metric in self.metric_specs:
            metric_id = self._resolve_metric_id(pynvml, metric.metric_id_names)
            if metric_id is None:
                continue
            metric_ids.append(metric_id)
            metric_names.append(self._metric_name(metric.name))

        if not metric_ids:
            if self.warn:
                warn_once(logger, "No requested NVML GPM metrics exist in pynvml.")
            return

        try:
            self._previous_sample = pynvml.nvmlGpmSampleAlloc()
            pynvml.nvmlGpmSampleGet(self._handle, self._previous_sample)
        except Exception as e:
            self._previous_sample = None
            if self.warn:
                warn_once(logger, f"Failed to start NVML GPM sampling: {e}")
            return

        self._gpm_metric_ids = metric_ids
        self._gpm_metric_names = metric_names
        self._gpm_supported = True

    def _get_gpm_metrics(self) -> dict[str, float]:
        assert self._pynvml is not None
        pynvml = self._pynvml

        current_sample = None
        try:
            current_sample = pynvml.nvmlGpmSampleAlloc()
            pynvml.nvmlGpmSampleGet(self._handle, current_sample)

            request = pynvml.c_nvmlGpmMetricsGet_t()
            request.version = pynvml.NVML_GPM_METRICS_GET_VERSION
            request.numMetrics = len(self._gpm_metric_ids)
            request.sample1 = self._previous_sample
            request.sample2 = current_sample
            for i, metric_id in enumerate(self._gpm_metric_ids):
                request.metrics[i].metricId = metric_id

            result = pynvml.nvmlGpmMetricsGet(request)
            metrics: dict[str, float] = {}
            success = getattr(pynvml, "NVML_SUCCESS", 0)
            for i, name in enumerate(self._gpm_metric_names):
                metric = result.metrics[i]
                value = float(metric.value)
                if (
                    getattr(metric, "nvmlReturn", success) == success
                    and math.isfinite(value)
                ):
                    metrics[name] = value

            return metrics
        except Exception as e:
            if self.warn:
                warn_once(logger, f"Failed to collect NVML GPM metrics: {e}")
            return {}
        finally:
            if current_sample is not None:
                if self._previous_sample is not None:
                    try:
                        pynvml.nvmlGpmSampleFree(self._previous_sample)
                    except Exception:
                        pass
                self._previous_sample = current_sample

    def _metric_name(self, name: str) -> str:
        return f"{self.prefix}{name}" if self.prefix else name

    def _resolve_metric_id(
        self, pynvml: ModuleType, metric_id_names: str | tuple[str, ...]
    ) -> int | None:
        if isinstance(metric_id_names, str):
            metric_id_names = (metric_id_names,)

        for metric_id_name in metric_id_names:
            metric_id = getattr(pynvml, metric_id_name, None)
            if isinstance(metric_id, int):
                return metric_id
        return None


def build_nvml_gpu_metrics_monitor(
    *,
    enabled: bool = True,
    device_index: int | None = None,
    enable_gpm: bool = True,
    prefix: str = "",
    warn: bool = True,
) -> NvmlGpuMetricsMonitor | None:
    if not enabled:
        return None
    return NvmlGpuMetricsMonitor(
        device_index=device_index,
        enable_gpm=enable_gpm,
        prefix=prefix,
        warn=warn,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Log NVIDIA GPU metrics from nvidia-ml-py/NVML as JSON lines."
    )
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=0, help="0 means run forever")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--disable-gpm", action="store_true")
    args = parser.parse_args()

    with NvmlGpuMetricsMonitor(
        device_index=args.device_index,
        enable_gpm=not args.disable_gpm,
        prefix=args.prefix,
    ) as monitor:
        sample = 0
        while args.samples <= 0 or sample < args.samples:
            metrics = monitor.get_metrics()
            print(json.dumps({"time": time.time(), "metrics": metrics}), flush=True)
            sample += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    _main()
