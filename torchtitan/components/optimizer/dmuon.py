# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def load_dmuon():
    """Load the optional DMuon dependency, including the vendored submodule."""
    try:
        return importlib.import_module("dmuon")
    except ModuleNotFoundError as error:
        if error.name != "dmuon":
            raise
        source_root = Path(__file__).resolve().parents[3] / "third_party" / "dmuon"
        if not source_root.is_dir():
            raise ModuleNotFoundError(
                "DMuon is not installed. Install third_party/dmuon with "
                "'pip install -e third_party/dmuon'."
            ) from error
        sys.path.insert(0, str(source_root))
        return importlib.import_module("dmuon")


__all__ = ["load_dmuon"]
