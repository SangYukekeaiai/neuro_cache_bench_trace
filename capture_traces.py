#!/usr/bin/env python3
"""Capture binary input spikes for one or more workloads."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from benchmarks import (
    BLOCKED_WORKLOADS,
    BenchmarkBlocked,
    REGISTRY,
    SpikingFC,
    SpikingGEMM,
    _reset_spiking_state,
)

CHUNK_NUMERICAL_NOTE = (
    "Chunked capture may differ from single-pass capture due to floating-point "
    "nondeterminism from batch-size-dependent reduction order."
)
C01_CHUNK_CHECK_NOTE = (
    " Reference checks at n=5, chunk=2 only: CIFAR10-DVS worst-layer mismatch "
    "was 2.9296875e-4 of elements (approximately 2.9e-4); layers 1-2 were exact "
    "and firing rates were identical to 4 digits. DVS128-Gesture was bit-identical "
    "at n=5, chunk=2. These measurements do not bound n=500 differences: an "
    "unchunked n=500 reference is unavailable because of OOM. The n=500 validator "
    "pass is structural, not a numerical-equivalence check."
)


def _sanitize_module_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return sanitized or "unnamed"


def _is_target_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Linear, SpikingGEMM, SpikingFC))


def _target_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    targets: list[tuple[str, nn.Module]] = []
    custom_gemm_ids: set[int] = set()
    for name, module in model.named_modules():
        if isinstance(module, (SpikingGEMM, SpikingFC)):
            custom_gemm_ids.add(id(module))
        if not _is_target_module(module):
            continue
        # SpikingGEMM/SpikingFC own a weight-bearing nn.Linear, but the custom
        # module is the intended target and gives the [M,K] contract.
        if isinstance(module, nn.Linear):
            parent_name = name.rpartition(".")[0]
            parent = model.get_submodule(parent_name) if parent_name else model
            if id(parent) in custom_gemm_ids or isinstance(parent, (SpikingGEMM, SpikingFC)):
                continue
        targets.append((name, module))
    return targets


def _module_params(module: nn.Module) -> dict[str, int]:
    if isinstance(module, nn.Conv2d):
        if module.kernel_size[0] != module.kernel_size[1]:
            raise ValueError(f"target convolution is not square: {module}")
        if module.stride[0] != module.stride[1]:
            raise ValueError(f"target convolution is not square-stride: {module}")
        return {
            "cout": int(module.out_channels),
            "kernel": int(module.kernel_size[0]),
            "stride": int(module.stride[0]),
        }

    if isinstance(module, (nn.Linear, SpikingGEMM, SpikingFC)):
        return {"cout": int(module.out_features), "kernel": 1, "stride": 1}

    raise TypeError(f"unsupported target module {type(module).__name__}")


def _assert_binary(value: torch.Tensor, module_name: str) -> None:
    if not torch.all((value == 0) | (value == 1)).item():
        raise AssertionError(f"input to {module_name} is not exactly binary")


def _split_timestep_input(
    value: torch.Tensor, timestep_count: int, sample_count: int, module_name: str,
    explicit_steps: bool = False,
) -> list[torch.Tensor]:
    """Turn either [T*N,...], [T,N,...], or one [N,...] call into steps."""

    if value.ndim == 0:
        raise AssertionError(f"target layer {module_name} received scalar input")
    if explicit_steps and value.shape[:2] == (timestep_count, sample_count):
        return list(value.unbind(0))
    if value.shape[0] == timestep_count * sample_count:
        return list(value.reshape(timestep_count, sample_count, *value.shape[1:]).unbind(0))
    if value.ndim >= 2 and value.shape[0] == timestep_count and value.shape[1] == sample_count:
        return list(value.unbind(0))
    if value.shape[0] == sample_count:
        if timestep_count != 1:
            # This is the normal shape for a module called once per timestep;
            # the caller appends one returned item for each hook invocation.
            return [value]
        return [value]
    raise AssertionError(
        f"target layer {module_name} input has leading shape {tuple(value.shape[:2])}; "
        f"expected T*N={timestep_count * sample_count}, [T,N]=({timestep_count},{sample_count}), "
        f"or N={sample_count}"
    )


def _capture(
    model: nn.Module,
    samples: Any,
    sample_count: int,
    timestep_count: int,
    device: torch.device,
    allow_unfired: bool,
    *,
    compact: bool = False,
    first_calls: list[tuple[str, nn.Module, bool]] | None = None,
) -> tuple[list[tuple[str, nn.Module, list[np.ndarray]]], dict[str, str]]:
    captures: list[tuple[str, nn.Module, list[np.ndarray]]] = []
    records: dict[int, tuple[str, list[np.ndarray]]] = {}
    declared_targets = _target_modules(model)
    fired_targets: set[int] = set()
    excluded: dict[str, str] = {}
    next_layer_number = 1
    handles = []

    for module_name, module in declared_targets:
        def hook(
            hooked_module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            _output: Any,
            *,
            declaration_name: str = module_name,
            target_module: nn.Module = module,
        ) -> None:
            nonlocal next_layer_number
            del hooked_module
            first_call = id(target_module) not in fired_targets
            fired_targets.add(id(target_module))
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise AssertionError(f"target layer {declaration_name} received no tensor input")
            if declaration_name in excluded:
                return
            value = inputs[0].detach()
            # For singleton chunks, T*N equals T. Use the target's input rank
            # to recognize explicit [T,N,...] before the flattened form.
            explicit_steps = compact and (
                (isinstance(target_module, nn.Conv2d) and value.ndim == 5)
                or (isinstance(target_module, (nn.Linear, SpikingGEMM, SpikingFC)) and value.ndim >= 3)
            )
            values = _split_timestep_input(
                value, timestep_count, sample_count, declaration_name, explicit_steps
            )
            binary = all(torch.all((value == 0) | (value == 1)).item() for value in values)
            if first_call and first_calls is not None:
                first_calls.append((declaration_name, target_module, binary))
            if not binary:
                excluded[declaration_name] = (
                    "weight-bearing input is not exactly binary; excluded from capture"
                )
                # A layer can fire with binary input at an earlier timestep and
                # then receive a non-binary input later.  Drop the partial
                # record so exclusion is a real outcome rather than a delayed
                # timestep-count assertion over stale values.
                if records.pop(id(target_module), None) is not None:
                    for index, (_name, captured_module, _values) in enumerate(captures):
                        if captured_module is target_module:
                            del captures[index]
                            break
                return

            record = records.get(id(target_module))
            if record is None:
                layer_number = next_layer_number
                next_layer_number += 1
                name = f"layer_{layer_number:02d}_{_sanitize_module_name(declaration_name)}"
                arrays: list[np.ndarray] = []
                records[id(target_module)] = (name, arrays)
                captures.append((name, target_module, arrays))
            else:
                name, arrays = record
            for value in values:
                _assert_binary(value, name)
                arrays.append(value.to(device="cpu", dtype=torch.uint8 if compact else value.dtype).numpy().copy())

        handles.append(module.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            model(_move_to_device(samples, device))
    finally:
        for handle in handles:
            handle.remove()

    if not allow_unfired:
        silent_modules = [
            module_name
            for module_name, module in declared_targets
            if id(module) not in fired_targets
        ]
        if silent_modules:
            raise AssertionError(
                "target modules did not fire during forward: " + ", ".join(silent_modules)
            )

    for layer_name, _module, values in captures:
        if len(values) != timestep_count:
            raise AssertionError(
                f"{layer_name} fired {len(values)} times; expected {timestep_count}"
            )

    return captures, excluded


def _capture_chunked(
    model: nn.Module,
    dataset: Any,
    sample_indices: list[int],
    chunk_size: int,
    timestep_count: int,
    device: torch.device,
    allow_unfired: bool,
) -> tuple[list[tuple[str, nn.Module, list[np.ndarray]]], dict[str, str]]:
    pieces: dict[nn.Module, list[list[np.ndarray]]] = {}
    first_calls: list[tuple[str, nn.Module, bool]] = []
    excluded: dict[str, str] = {}
    for start in range(0, len(sample_indices), chunk_size):
        _reset_spiking_state(model)
        indices = sample_indices[start:start + chunk_size]
        samples = _stack_samples([dataset[index] for index in indices])
        calls: list[tuple[str, nn.Module, bool]] = []
        captures, chunk_excluded = _capture(
            model, samples, len(indices), timestep_count, device, allow_unfired,
            compact=True, first_calls=calls,
        )
        if not start:
            first_calls = calls
        else:
            if [module for _, module, _ in calls] != [module for _, module, _ in first_calls]:
                raise AssertionError("target firing order differs between sample chunks")
            first_calls = [
                (name, module, binary and current_binary)
                for (name, module, binary), (_, _, current_binary) in zip(first_calls, calls)
            ]
        excluded.update(chunk_excluded)
        for name, module, _ in calls:
            if name in excluded:
                pieces.pop(module, None)
        for _, module, values in captures:
            name = next(name for name, target, _ in calls if target is module)
            if name not in excluded:
                if isinstance(module, (nn.Linear, SpikingGEMM, SpikingFC)):
                    values = [value[:, None, :] if value.ndim == 2 else value for value in values]
                pieces.setdefault(module, []).append(values)
        print(f"captured chunk samples {start + 1}-{start + len(indices)}/{len(sample_indices)}")

    merged = []
    layer_number = 0
    for name, module, first_binary in first_calls:
        # A full-batch hook allocates a layer number only if its first call is
        # binary across every sample; a later exclusion still consumes it.
        if first_binary:
            layer_number += 1
        if name in excluded:
            continue
        values = [
            np.concatenate([chunk[timestep] for chunk in pieces[module]], axis=0)
            for timestep in range(timestep_count)
        ]
        merged.append((f"layer_{layer_number:02d}_{_sanitize_module_name(name)}", module, values))
    return merged, excluded


def _stack_layer(
    layer_name: str,
    module: nn.Module,
    per_timestep: list[np.ndarray],
    sample_count: int,
) -> tuple[np.ndarray, list[int], dict[str, int] | None]:
    first = per_timestep[0]
    if isinstance(module, nn.Conv2d):
        if first.ndim != 4:
            raise ValueError(f"{layer_name} convolution input has shape {first.shape}")
        if first.shape[0] != sample_count:
            raise ValueError(f"{layer_name} batch dimension is {first.shape[0]}, expected {sample_count}")
        array = np.stack(per_timestep, axis=0).astype(np.uint8, copy=False)
        return array, [int(value) for value in array.shape], None

    if isinstance(module, (nn.Linear, SpikingGEMM, SpikingFC)):
        if first.ndim == 2:
            per_timestep = [value[:, None, :] for value in per_timestep]
            first = per_timestep[0]
        if first.ndim != 3:
            raise ValueError(f"{layer_name} GEMM input has shape {first.shape}")
        if first.shape[0] != sample_count:
            raise ValueError(f"{layer_name} batch dimension is {first.shape[0]}, expected {sample_count}")
        array = np.stack(per_timestep, axis=0).transpose(0, 1, 3, 2)[:, :, :, :, None]
        array = array.astype(np.uint8, copy=False)
        return array, [int(value) for value in array.shape], {
            "M": int(first.shape[1]),
            "K": int(first.shape[2]),
            "N": int(module.out_features),
        }

    raise TypeError(f"unsupported target module {type(module).__name__}")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _stack_samples(samples: Sequence[Any]) -> Any:
    first = samples[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(list(samples), dim=0)
    if isinstance(first, Mapping):
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in first}
    return torch.stack([torch.as_tensor(sample) for sample in samples], dim=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--n", type=int, required=True, help="number of unique samples to capture")
    parser.add_argument("--seed", type=int, help="numpy subset-selection seed")
    parser.add_argument(
        "--sample-start",
        type=int,
        help="select the contiguous dataset-order range starting at this index",
    )
    parser.add_argument("--out", type=Path, required=True, help="parent output directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--chunk", type=int, help="samples per forward pass; default captures all samples at once")
    parser.add_argument(
        "--allow-unfired",
        action="store_true",
        help="allow declared target modules that do not fire during the forward pass",
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.seed is None and args.sample_start is None:
        parser.error("--seed is required unless --sample-start is provided")
    if args.sample_start is not None and args.sample_start < 0:
        parser.error("--sample-start must be non-negative")
    if args.chunk is not None and args.chunk <= 0:
        parser.error("--chunk must be positive")
    return args


def _workload_list(loaded: Any) -> list[tuple[nn.Module, Any, int, str, str]]:
    if isinstance(loaded, tuple) and len(loaded) == 5:
        return [loaded]
    if isinstance(loaded, list):
        return loaded
    raise TypeError(f"registry loader returned unsupported value {type(loaded)}")


def _capture_workload(
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    dataset: Any,
    timestep_count: int,
    arch_name: str,
    dataset_name: str,
) -> Path:
    selected_dataset = dataset.choose(args.n) if hasattr(dataset, "choose") else dataset
    sample_start = getattr(args, "sample_start", None)
    seed = getattr(args, "seed", None)
    if sample_start is not None:
        if sample_start < 0:
            raise ValueError("--sample-start must be non-negative")
        if sample_start + args.n > len(selected_dataset):
            raise ValueError(
                f"--sample-start={sample_start} with --n={args.n} exceeds dataset size "
                f"{len(selected_dataset)}"
            )
        sample_indices = list(range(sample_start, sample_start + args.n))
        selected_by = (
            f"contiguous dataset-order range [{sample_start}, {sample_start + args.n})"
        )
    else:
        if args.n > len(selected_dataset):
            raise ValueError(f"--n={args.n} exceeds dataset size {len(selected_dataset)}")
        if seed is None:
            raise ValueError("--seed is required unless --sample-start is provided")
        rng = np.random.default_rng(seed)
        sample_indices = sorted(int(index) for index in rng.choice(len(selected_dataset), args.n, replace=False))
        selected_by = "numpy.random.default_rng(seed).choice(len(selected_dataset), n, replace=False), sorted"

    model = model.to(device=device).eval()

    if args.chunk is None:
        samples = _stack_samples([selected_dataset[index] for index in sample_indices])
        captures, excluded_layers = _capture(
            model, samples, len(sample_indices), timestep_count, device,
            allow_unfired=args.allow_unfired,
        )
    else:
        captures, excluded_layers = _capture_chunked(
            model, selected_dataset, sample_indices, args.chunk, timestep_count,
            device, allow_unfired=args.allow_unfired,
        )
    output_dir = args.out / f"{arch_name}_T{timestep_count}_n{args.n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_device = next(model.parameters()).device
    actual_device_text = str(actual_device)
    if actual_device.type == "cuda":
        actual_device_text += f" ({torch.cuda.get_device_name(actual_device)})"

    layers: dict[str, list[int]] = {}
    layer_params: dict[str, dict[str, int]] = {}
    gemm_meta: dict[str, dict[str, int]] = {}
    for layer_name, module, per_timestep in captures:
        array, shape, observed_gemm = _stack_layer(
            layer_name, module, per_timestep, len(sample_indices)
        )
        if observed_gemm is not None:
            gemm_meta[layer_name] = observed_gemm
        layers[layer_name] = shape
        layer_params[layer_name] = _module_params(module)
        np.save(output_dir / f"{layer_name}.npy", array)

    selected_name = getattr(selected_dataset, "trace_split_name", dataset_name)
    meta: dict[str, Any] = {
        "arch": arch_name,
        "batch_size": len(sample_indices),
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": selected_name,
        "device": {
            "requested": args.device,
            "actual": actual_device_text,
        },
        "dtype": "uint8",
        "layer_params": layer_params,
        "layers": layers,
        "n_samples": len(sample_indices),
        "subset": {
            "of": f"{arch_name}_T{timestep_count}_full",
            "n": len(sample_indices),
            "seed": seed,
            "sample_indices": sample_indices,
            "selected_by": selected_by,
            "made_by": "capture_traces.py",
        },
        "timestep": timestep_count,
    }
    if sample_start is not None:
        meta["subset"]["sample_start"] = sample_start
    if args.chunk is not None:
        meta["chunk_size"] = args.chunk
        meta["numerical_note"] = CHUNK_NUMERICAL_NOTE
        if args.benchmark == "C01":
            meta["numerical_note"] += C01_CHUNK_CHECK_NOTE
    if gemm_meta:
        meta["gemm"] = gemm_meta
    if excluded_layers:
        meta["excluded_layers"] = excluded_layers

    with (output_dir / "meta.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(meta, metadata_file, indent=2)
        metadata_file.write("\n")

    print(
        f"captured {len(captures)} layers ({len(excluded_layers)} excluded) for "
        f"{len(sample_indices)} samples in {output_dir}"
    )
    print(f"sample_indices={sample_indices}")
    return output_dir


def main() -> int:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)

    try:
        loaded = REGISTRY[args.benchmark]()
    except BenchmarkBlocked as exc:
        print(f"BLOCKED {args.benchmark}: {exc}")
        return 0

    workloads = _workload_list(loaded)
    for model, dataset, timestep_count, arch_name, dataset_name in workloads:
        if not model.training:
            model.eval()
        if timestep_count <= 0:
            raise ValueError(f"benchmark returned invalid timestep count {timestep_count}")
        _capture_workload(
            args,
            device,
            model,
            dataset,
            timestep_count,
            arch_name,
            dataset_name,
        )

    for blocked in BLOCKED_WORKLOADS:
        print(f"BLOCKED {args.benchmark}: {blocked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
