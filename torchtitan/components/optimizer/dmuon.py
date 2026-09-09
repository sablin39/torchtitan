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
    """Load the optional DMuon dependency, preferring the vendored submodule.

    The vendored fork is the pinned source of truth; an installed dmuon copy
    is only a fallback. Checking it first keeps a stale site-packages install
    from silently shadowing fork changes.
    """
    source_root = Path(__file__).resolve().parents[3] / "third_party" / "dmuon"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        return importlib.import_module("dmuon")
    except ModuleNotFoundError as error:
        if error.name != "dmuon":
            raise
        raise ModuleNotFoundError(
            "DMuon is not installed. Install third_party/dmuon with "
            "'pip install -e third_party/dmuon'."
        ) from error


__all__ = ["load_dmuon"]
