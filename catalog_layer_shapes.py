"""Write LAYER_SHAPES.md: the layer shapes of every captured trace.

Shapes come from each trace's meta.json ("layers", "layer_params", "gemm",
"timestep", "n_samples"). Conv padding, dilation and groups are not in
meta.json. Metadata-only mode uses explicit settings for the pinned architectures;
otherwise they are read from models built by benchmarks.py (CPU, no forward pass).

Run: conda run -n base .venv/bin/python -I catalog_layer_shapes.py --metadata-only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # python -I drops the script dir from sys.path
TRACE_ROOTS = ("out_real", "out", "out_second")
DELTA_ROOT = ROOT.parent / "examples/input_trace/delta"
OUT_MD = ROOT / "LAYER_SHAPES.md"

# kernel -> (padding, dilation, groups), module-derived for the pinned models.
# C01 settings are checked statically in the VGG8 constructors.
CONV_SETTINGS = {
    "spikformer_4_384": {3: (1, 1, 1)},
    "sdt_2_512": {1: (0, 1, 1), 3: (1, 1, 1)},
    "vgg16_cifar10_loas": {3: (1, 1, 1)},
    "loas_resnet19": {1: (0, 1, 1), 3: (1, 1, 1)},
    "loas_vgg16": {3: (1, 1, 1)},
    "sma_azo_vgg_cifar10_dvs": {3: (1, 1, 1)},
    "sma_azo_vgg_dvs128_gesture": {3: (1, 1, 1)},
}


def trace_dirs(roots) -> list[tuple[str, pathlib.Path]]:
    """(registry id, trace dir) for every meta.json under the selected roots."""
    found = []
    for root in roots:
        for meta_path in sorted((ROOT / root).glob("**/meta.json")):
            rel = meta_path.parent.relative_to(ROOT / root)
            bench = rel.parts[0] if len(rel.parts) > 1 else "synthetic"
            found.append((bench, meta_path.parent))
    return found


def load_modules(bench: str) -> dict[str, dict]:
    """arch name -> {sanitized module name -> module} for one registry entry."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import benchmarks
    from capture_traces import _sanitize_module_name, _target_modules

    loaded = benchmarks.REGISTRY[bench]()
    workloads = loaded if isinstance(loaded, list) else [loaded]
    return {arch: {_sanitize_module_name(n): m for n, m in _target_modules(model)}
            for model, _ds, _t, arch, _dsname in workloads}


def conv_out(size: int, k: int, s: int, p: int, d: int) -> int:
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def layer_rows(meta: dict, modules: dict | None) -> list[dict]:
    if modules is not None:
        from torch import nn
    rows = []
    gemm = meta.get("gemm", {})
    for name, (t, batch, cin, hin, win) in meta["layers"].items():
        assert (t, batch) == (meta["timestep"], meta["n_samples"]), name
        p = meta["layer_params"][name]
        module = modules.get(name.split("_", 2)[2]) if modules is not None else None
        row = {"name": name, "t": t, "inc": cin, "oc": p["cout"], "hin": hin, "win": win,
               "stride": p["stride"]}
        if name in gemm:
            g = gemm[name]
            assert (g["K"], g["M"], g["N"], win) == (cin, hin, p["cout"], 1), name
            row.update(type="spiking-GEMM", kh=1, kw=1, ho=g["M"], wo=1,
                       padding=0, dilation=1, groups=1, M=g["M"], K=g["K"], N=g["N"])
        else:
            k, s = p["kernel"], p["stride"]
            row.update(type="spiking-Conv", kh=k, kw=k, M="", K="", N="")
            if modules is None:
                settings = CONV_SETTINGS.get(meta["arch"], {})
                strides = (1, 2) if meta["arch"] == "loas_resnet19" else (1,)
                if k not in settings or s not in strides:
                    raise ValueError(f"No verified conv settings for {meta['arch']}/{name}: kernel={k}, stride={s}")
                pad, dil, groups = settings[k]
                row.update(padding=pad, dilation=dil, groups=groups,
                           ho=conv_out(hin, k, s, pad, dil), wo=conv_out(win, k, s, pad, dil))
            elif isinstance(module, nn.Conv2d) and module.padding in {(q, q) for q in range(k)} \
                    and module.dilation[0] == module.dilation[1]:
                assert (module.out_channels, module.kernel_size[0], module.stride[0]) == (p["cout"], k, s), name
                pad, dil = module.padding[0], module.dilation[0]
                row.update(padding=pad, dilation=dil, groups=module.groups,
                           ho=conv_out(hin, k, s, pad, dil), wo=conv_out(win, k, s, pad, dil))
            else:
                row.update(padding="unknown", dilation="unknown", groups="unknown",
                           ho="unknown", wo="unknown")
        rows.append(row)
    return rows


COLS = ["#", "layer", "type", "kh", "kw", "oc", "inc", "ho", "wo", "t",
        "hin", "win", "stride", "padding", "dilation", "groups", "M", "K", "N"]


def table(rows: list[dict]) -> list[str]:
    out = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for i, r in enumerate(rows, 1):
        cells = [i, f"`{r['name']}`"] + [r[c] for c in COLS[2:]]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, help="Trace root containing registry ID directories")
    parser.add_argument("--metadata-only", action="store_true", help="Read metadata without loading models; default root is delta")
    args = parser.parse_args()
    roots = (args.root.absolute(),) if args.root else ((DELTA_ROOT,) if args.metadata_only else TRACE_ROOTS)
    entries = []  # (bench, dir, meta, rows)
    module_cache: dict[str, dict] = {}
    for bench, d in trace_dirs(roots):
        meta = json.loads((d / "meta.json").read_text())
        if not args.metadata_only and bench not in module_cache:
            module_cache[bench] = load_modules(bench)
        rows = layer_rows(meta, None if args.metadata_only else module_cache[bench].get(meta["arch"], {}))
        entries.append((bench, d, meta, rows))
    if not entries:
        raise ValueError(f"No meta.json files found under {roots}")
    primary = [e for e in entries if e[0] != "synthetic"]
    others = [e for e in entries if e not in primary]
    sample_counts = ', '.join(str(n) for n in sorted({e[2]['n_samples'] for e in primary}))
    seeds = ', '.join(str(n) for n in sorted({e[2]['subset']['seed'] for e in primary}))
    root_text = ', '.join(f'`{ROOT / r}`' for r in roots)
    regen = "conda run -n base .venv/bin/python -I catalog_layer_shapes.py"
    if args.metadata_only:
        regen += " --metadata-only"
    if args.root:
        import shlex
        regen += " --root " + shlex.quote(str(args.root.absolute()))

    md = [
        "# Layer shapes of the captured input spike traces",
        "",
        f"Generated by `catalog_layer_shapes.py` (`{regen}`); "
        "do not edit by hand.",
        "",
        "## Conventions",
        "",
        "- Every trace `.npy` is binary `uint8` with shape `[T, N, Cin, Hin, Win]`, recorded in "
        "`meta.json` `layers`. `t` = T, `inc` = Cin, `oc` = `layer_params.cout`, "
        "`kh = kw = layer_params.kernel`, `stride = layer_params.stride`.",
        "- **spiking-GEMM** = an `nn.Linear` (or `SpikingGEMM`/`SpikingFC`) target, i.e. a layer "
        "listed in `meta.json` `gemm`. Its input `[M, K]` is stored as `Hin = M`, `Win = 1`, "
        "`Cin = K`; output features `N = cout`. In this table a GEMM is written as a 1x1, stride-1, "
        "unpadded conv: `kh = kw = 1`, `inc = K`, `oc = N`, `ho = M`, `wo = 1`. "
        "For transformer layers M is the token count per sample.",
        "- **spiking-Conv** = an `nn.Conv2d` target (not in `gemm`). `ho`/`wo` use the module's "
        "real padding and dilation: `ho = (Hin + 2*padding - dilation*(kh-1) - 1) // stride + 1`. "
        "Padding, dilation and groups for A02–B09 are module-derived and were not re-verified "
        "against the delta metadata, which does not contain these fields. For C01, the captured "
        "convolutions use padding 1, dilation 1 and groups 1, checked from the pinned VGG8 "
        "constructors without executing them. (The wcache loader `build_workload_from_trace` instead "
        "assumes `PAD = 1`, 3x3, stride 1 and skips every other layer via `exclusion_reason`.)",
        "- `#` is the capture order (`layer_NN` prefix), which is the order the layer first fired "
        "in the forward pass, i.e. network order.",
        "- Weight-bearing layers whose input was not exactly 0/1 were not captured; they are listed "
        "per benchmark under \"Not captured\" and have no trace, so no shape row.",
        "",
        "## Benchmarks",
        "",
        "| registry id | model (arch) | dataset split | T | N | layers | GEMM | Conv | capture directory |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for bench, d, meta, rows in entries:
        c = Counter(r["type"] for r in rows)
        md.append(f"| {bench} | `{meta['arch']}` | `{meta['dataset']}` | {meta['timestep']} | "
                  f"{meta['n_samples']} | {len(rows)} | {c['spiking-GEMM']} | {c['spiking-Conv']} | "
                  f"`{d}` |")
    md.append("")
    for bench, _d, meta, _rows in entries:
        if "chunk_size" in meta:
            md += [f"{bench}/`{meta['arch']}` metadata records `chunk_size: {meta['chunk_size']}`. "
                   + meta.get("numerical_note", ""), ""]
    if others:
        md += ["`out/` and `out_second/` hold only the `synthetic` registry entry (the local smoke "
               "benchmark: seeded random weights and synthetic spikes, no upstream model); the two "
               "copies differ only in `captured_at`. It is listed for completeness and left out of the "
               "cross-benchmark summary.", ""]

    md += ["## Per-benchmark layer tables", ""]
    for bench, d, meta, rows in entries:
        md += [f"### {bench} / `{meta['arch']}`", "",
               f"`{d}` - dataset `{meta['dataset']}`, T = {meta['timestep']}, "
               f"N = {meta['n_samples']}, {len(rows)} captured layers.", ""]
        md += table(rows) + [""]
        excl = meta.get("excluded_layers", {})
        if excl:
            md += [f"Not captured ({len(excl)}): " + ", ".join(f"`{k}`" for k in excl)
                   + f". Reason recorded: {sorted(set(excl.values()))}.", ""]

    key = ("kh", "kw", "oc", "inc", "ho", "wo", "t")
    classes: dict[tuple, Counter] = {}
    types: dict[tuple, str] = {}
    for bench, _d, meta, rows in primary:
        for r in rows:
            k = tuple(r[c] for c in key)
            classes.setdefault(k, Counter())[f"{bench}/{meta['arch']}"] += 1
            types[k] = r["type"]
    md += ["## Cross-benchmark summary", "",
           f"Over the n={sample_counts} delta captures: {sum(len(e[3]) for e in primary)} captured layers in "
           f"{len(primary)} workloads, {len(classes)} distinct (kh, kw, oc, inc, ho, wo, t) shape "
           "classes.", "",
           "### GEMM vs Conv per workload", "",
           "| workload | spiking-GEMM | spiking-Conv | total |", "|---|---:|---:|---:|"]
    for bench, _d, meta, rows in primary:
        c = Counter(r["type"] for r in rows)
        md.append(f"| {bench}/`{meta['arch']}` | {c['spiking-GEMM']} | {c['spiking-Conv']} | {len(rows)} |")
    md += ["", "### Shape classes", "",
           "| type | kh | kw | oc | inc | ho | wo | t | count | workloads (count) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, cnt in sorted(classes.items(), key=lambda kv: (types[kv[0]], -sum(kv[1].values()), str(kv[0]))):
        where = ", ".join(f"{w} ({n})" for w, n in sorted(cnt.items()))
        md.append(f"| {types[k]} | " + " | ".join(str(v) for v in k) + f" | {sum(cnt.values())} | {where} |")

    md += ["", "## How this was derived", "",
           f"- Trace directories: all {len(entries)} `meta.json` files under {root_text}. "
           f"The tables cover the real n={sample_counts} captures.",
           "- From `meta.json`: `arch`, `dataset`, `timestep` (T), `n_samples` (N); "
           "`layers[name] = [T, N, Cin, Hin, Win]`; `layer_params[name] = {cout, kernel, stride}`; "
           "`gemm[name] = {M, K, N}` (presence marks a GEMM; K = Cin, M = Hin, "
           f"N = cout, Win = 1); `excluded_layers`. Every trace has N = {sample_counts} and subset seed {seeds}.",
           "- These fields are written by `capture_traces.py` (`_capture_workload`, `_stack_layer`, "
           "`_module_params`). Layer names are `layer_NN_<sanitized module name>`.",
           "- Conv padding, dilation and groups for A02–B09 are module-derived and were not "
           "re-verified against the delta metadata. C01 values were read statically from `VGG8` "
           "in `upstream/SMA-AZO/SMA VGG/vgg8_cifar10dvs_SMA.py` and `vgg8_dvs128_SMA.py`: "
           "block2–block5 use 3x3 stride-1 padding-1 convolutions with default dilation/groups "
           "of 1. Metadata-only mode uses these explicit architecture settings in `CONV_SETTINGS` "
           "and rejects unsupported convolution settings. Output spatial dimensions use the formula above.",
           f"- The trace format is read by `{ROOT.parent / 'src/lays/weight_trace_gen/trace.py'}` "
           "(`load_layer_trace`, `layer_params`).",
           f"- Regenerate: `{regen}`. Metadata-only mode reads metadata without importing "
           "model loaders or loading models, datasets or checkpoints.", ""]
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}: {len(primary)} primary workloads, "
          f"{sum(len(e[3]) for e in primary)} layers, {len(classes)} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
