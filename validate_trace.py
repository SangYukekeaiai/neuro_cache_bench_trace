#!/usr/bin/env python3
"""Validate a captured trace through the project's canonical trace loader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

DEFAULT_WCACHE_SRC = "/home/ya867177/neuro_cache-wcache/src"
WCACHE_SRC = os.environ.get("WCACHE_SRC") or DEFAULT_WCACHE_SRC
if WCACHE_SRC not in sys.path:
    sys.path.insert(0, WCACHE_SRC)

from lays.weight_trace_gen.trace import exclusion_reason, layer_params, load_layer_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()

    meta_path = args.trace_dir / "meta.json"
    try:
        with meta_path.open(encoding="utf-8") as metadata_file:
            meta = json.load(metadata_file)
        layer_names = list(meta["layers"])
    except Exception as exc:
        print(f"meta.json: FAIL: {exc}")
        return 1

    failed = False
    for layer_name in layer_names:
        try:
            trace = load_layer_trace(args.trace_dir, layer_name, mmap=True)
            params = layer_params(meta, layer_name)
        except Exception as exc:
            failed = True
            print(f"{layer_name}: FAIL: {exc}")
            continue

        warnings = []
        if trace.dtype != np.uint8:
            warnings.append(f"dtype is {trace.dtype}, expected uint8")
        if not np.all((trace == 0) | (trace == 1)):
            warnings.append("trace contains values other than 0 and 1")
        reason = exclusion_reason(meta, layer_name)
        if reason is not None:
            warnings.append(f"replay-excluded: {reason}")
        if warnings:
            print(f"{layer_name}: WARN shape={list(trace.shape)} params={params} "
                  f"({'; '.join(warnings)})")
        else:
            print(f"{layer_name}: PASS shape={list(trace.shape)} params={params}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
