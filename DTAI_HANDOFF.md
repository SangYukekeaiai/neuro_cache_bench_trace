# DeltaAI handoff: regenerating n=500 seed-0 captures without Slurm

This is the plain-server procedure for the DeltaAI GPU box. It mirrors the Delta
campaign documented in [slurm/README_DELTA.md](slurm/README_DELTA.md), with the
`sbatch` wrappers removed: on DeltaAI you run `capture_traces.py` directly and
supply the chunking and memory headroom yourself.

Run every step from the repository root unless a step says otherwise.

## Scope: A02 and A04 only

DeltaAI regenerates **A02 and A04 and nothing else.** The other six Nov 7 n=500
traces (A07, A08, two B09 workloads, two C01 workloads) remain valid and are
kept as they are; only A02/A04 were invalidated, by the switch to the CIFAR-100
loaders. Do not re-run them: an A07 capture alone costs 20G of output and ~80G
of host RAM to replace an already-validated artifact.

The directories to produce are:

| Benchmark | Directory (CIFAR-100) |
|---|---|
| A02 | `A02/spikformer_4_384_cifar100_T4_n500` |
| A04 | `A04/sdt_2_512_cifar100_T4_n500` |

These are new names, not replacements: the CIFAR-100 loaders tag the
architecture differently (`benchmarks.py:244,271`), so a CIFAR-100 capture can
never overwrite a CIFAR-10 one. There is no CIFAR-10 A02/A04 directory left to
collide with in any case — commit `fbae5bb` renamed those captures to the
`*_cifar100_*` names along with the loader fix, so each architecture now has
exactly one n=500 directory.

**As of 2026-11-10 both captures already exist and are mirrored**, made on a
GH200 node (`meta.json` `captured_at` 02:19:52Z for A02 and 03:54:16Z for A04,
device `cuda:0 (NVIDIA GH200 120GB)`), and both pass the T16 audit in section 7.
Nothing is queued on Delta for these two benchmarks: job `23689507` never ran
(`data/capture-A02-23689507.log` holds only
`slurm/capture_interactive.sbatch: Permission denied`), and the captures that
landed came by another route. There is no second machine to race.

**Run the presence check first.** Today it reports `PRESENT` for all four
paths, so step 5 is skipped; run it again on a fresh machine, or if a directory
is ever lost (`$OUT_ROOT` is set in step 1; do steps 1-4 first if this is a
fresh machine):

```bash
for d in /work/nvme/bebv/yyu9/input_trace_delta/A02/spikformer_4_384_cifar100_T4_n500 \
         /work/nvme/bebv/yyu9/input_trace_delta/A04/sdt_2_512_cifar100_T4_n500 \
         "$OUT_ROOT/A02/spikformer_4_384_cifar100_T4_n500" \
         "$OUT_ROOT/A04/sdt_2_512_cifar100_T4_n500"; do
  if [ -f "$d/meta.json" ]; then
    echo "PRESENT  $d  -- skip this capture"
  else
    echo "MISSING  $d  -- capture it"
  fi
done
```

The first two paths are the shared `/work/nvme` mirror, which Delta writes to
directly, so a finished Delta run shows up there without any transfer.

A directory without `meta.json` is a partial capture, not a finished one: treat
it as missing, remove it, and capture again. If both are `PRESENT` — which is
the expected result today — skip step 5 entirely and go straight to the step 6
check.

## 1. Repository and wcache paths

```bash
git clone git@github.com:SangYukekeaiai/neuro_cache_bench_trace.git
cd neuro_cache_bench_trace
# or, in an existing checkout:
git pull origin main
```

The checkout must be on a commit that contains the CIFAR-100 loader state for
A02/A04 (`_cifar` helper, `cifar100` stats, `ckpts/A0X/cifar100` paths,
`*_cifar100` model tags). Confirm before capturing:

```bash
grep -n 'cifar100' benchmarks.py | head
```

If that prints nothing the checkout predates the fix and the A02/A04 traces it
produces will not match the campaign.

Two paths outside this repository are expected:

- `WCACHE_SRC` — the `src/` directory of the `neuro_cache-wcache` checkout.
  `validate_trace.py` imports the canonical loader from it
  (`lays.weight_trace_gen.trace`). Its compiled-in default is
  `/home/ya867177/neuro_cache-wcache/src`, which is wrong on DeltaAI, so export
  the real path before validating.
- the capture output root. On Delta this was
  `$WCACHE/examples/input_trace/delta/$B`; on DeltaAI pick any writable directory
  and keep the per-benchmark `$B` component, because it is what stops two
  benchmarks with the same `<arch>_T<T>_n<N>` directory name from colliding.

- the **shared mirror**, `/work/nvme/bebv/yyu9/input_trace_delta/`. This is on
  `/work/nvme`, which is "Shared between Delta and DeltaAI"
  ([data management][data]), so both machines read and write the same
  directory. It already holds the six Nov 7 traces of section 7, plus the two
  A02/A04 captures. Every new capture gets
  copied into it (step 5), **add-only** — never delete or overwrite what is
  already there.

```bash
export WCACHE=/path/to/neuro_cache-wcache
export WCACHE_SRC="$WCACHE/src"
export OUT_ROOT="$WCACHE/examples/input_trace/deltaai"
export SHARED=/work/nvme/bebv/yyu9/input_trace_delta
```

Clone into `/work`, not `/u`: HOME is per-system (Delta's `/u` is not visible
here) and NCSA says it is "**Not** intended as a source or destination for I/O
during jobs" ([data management][data]).

`data/`, `ckpts/*` (except `CKPTS.md`), `upstream/*` (except `INVENTORY.md`),
`out*/`, and `*.log` are gitignored, so none of the checkpoints, datasets, or
traces arrive with the clone. Step 3 stages the checkpoints; step 4 only
verifies shared paths, because the datasets are already on `/work/nvme`.

## 2. Environment

DeltaAI is **not** Delta. Its compute nodes are 4-way NVIDIA GH200 Grace Hopper
superchips: a 72-core **ARM** Grace CPU and an H100 with 96 GB HBM3 per
superchip, four per node, 288 cores and 480 GB of CPU RAM per node
([architecture][arch]). The x86 `.venv` used on Delta will not run here, and
neither will anything else built for x86: NCSA states that Delta and DeltaAI
"have different home directories; the CPUs are different, incompatible
architectures" ([data management][data]). Build a fresh aarch64 environment on
DeltaAI; do not copy, reuse, or point at Delta's `.venv`, and do not expect
anything under Delta's `/u` to be visible.

Because `/u` is per-system and NCSA says HOME is "**Not** intended as a source
or destination for I/O during jobs" ([data management][data]), put the clone,
the environment, the checkpoints, and the output under your `/work` allocation
rather than home.

The package set to reproduce (the pinned versions are what Delta ran; the torch
build necessarily differs, because Delta's `+cu132` x86 wheels have no aarch64
equivalent):

```text
numpy==2.5.4              spikingjelly==0.0.0.0.14
timm==0.6.13              braincog==git f9b879f75da2247a9f0c31864a947f0e5d2f3dab
transformers==4.57.0      datasets==3.3.2
safetensors==0.8.0        accelerate==1.15.0
huggingface_hub==0.36.2
torch, torchvision        -- aarch64 / GH200 build, see below
```

Create the environment with an **aarch64** conda distribution — a
`Miniforge3-Linux-aarch64` installer, or whatever conda/python module DeltaAI
exposes — then install a **GH200-compatible** torch before anything else, and
confirm the wheel is really aarch64 + CUDA before continuing:

```bash
conda create -y -n neurocache python=3.12
conda activate neurocache
# install the aarch64 CUDA torch build for GH200 first, then the rest
python -m pip install --upgrade pip
python -m pip install torch torchvision          # must resolve to an aarch64 CUDA build
python -c "import platform, torch; print(platform.machine(), torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

That last line is the gate. It must print `aarch64`, a CUDA-enabled torch
version, `True`, and an H100/GH200 device name. If it prints `False` or no
device, stop: the rest of this document assumes `--device cuda` works.

**Not determined here:** the exact aarch64 torch install command for DeltaAI.
Neither NCSA page cited above mentions conda, Miniforge, module names, a
PyTorch index URL, or a GH200 container, so there is no sourced command to
quote. Find it from DeltaAI's own software/environment documentation or NGC
before running the install above, rather than assuming the default PyPI index
serves a working GH200 wheel.

With torch in place, the remaining packages install normally — they are pure
Python or build from source:

```bash
python -m pip install \
  spikingjelly==0.0.0.0.14 timm==0.6.13 transformers==4.57.0 \
  datasets==3.3.2 safetensors==0.8.0 accelerate==1.15.0
python -m pip install git+https://github.com/BrainCog-X/Brain-Cog.git@f9b879f75da2247a9f0c31864a947f0e5d2f3dab
```

Delta ran its venv interpreter through `conda run --no-capture-output -n base`
purely to get the site CUDA runtime onto the library path. On DeltaAI the conda
environment above is the environment, so invoke `python` from it directly.

One required patch, only if you ever rebuild the C01 event data (A02/A04 do not
touch it): SpikingJelly 0.0.0.0.14 calls `np.fromstring` when converting
CIFAR10-DVS events, which NumPy 2 removed. Without the patch `events_np/` stays
empty and loading fails with `FileNotFoundError: Found no valid file for the
classes`.

```bash
SJ=$(python -c "import spikingjelly, pathlib; print(pathlib.Path(spikingjelly.__file__).parent)")
sed -i "s/np.fromstring(data, dtype='>u4')/np.frombuffer(data, dtype='>u4')/" "$SJ/datasets/cifar10_dvs.py"
grep -n 'frombuffer' "$SJ/datasets/cifar10_dvs.py"
```

Thread caps. Delta used 16; a DeltaAI superchip has 72 cores, so size this to
what your job actually holds:

```bash
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
```

Then run the capture-hook regression test before any real capture. It is the
first thing that will fail if the aarch64 torch build is wrong:

```bash
python test_capture_traces.py
```

[arch]: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html
[data]: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html

## 3. Checkpoints

`ckpts/CKPTS.md` is the manifest and is the only thing under `ckpts/` that is
committed. Stage each file at the path the table names, then verify the digest
— the manifest's `bytes` and `sha256` columns are the acceptance test, and a
mismatch most often means an HTML error page was saved instead of the weights.

Hugging Face files, including the two CIFAR-100 checkpoints A02/A04 now load:

```bash
mkdir -p ckpts/A02/cifar100 ckpts/A04/cifar100 ckpts/A07

curl -L -o ckpts/A02/cifar100/model_best.pth.tar \
  https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/Spikformer/cifar100/model_best.pth.tar
curl -L -o ckpts/A04/cifar100/model_best.pth.tar \
  https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/SDT/cifar100/model_best.pth.tar

for f in model.safetensors config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json vocab.txt; do
  curl -L -o "ckpts/A07/$f" "https://huggingface.co/hamingsi/SpikingLM/resolve/main/$f"
done
```

B09 (and A08, which reuses B09's VGG-16) are copied out of the committed
upstream clone rather than downloaded:

```bash
mkdir -p ckpts/B09
cp upstream/LoAS/sample_ckpts/vgg16_final_dict.pth.tar \
   upstream/LoAS/sample_ckpts/resnet19_final_dict.pth.tar \
   upstream/LoAS/sample_ckpts/alexnet_final_dict.pth.tar ckpts/B09/
```

C01's two SMA-AZO-VGG checkpoints come from the Google Drive links in
`upstream/SMA-AZO/README.md`, fetched with `gdown` (the manifest records their
digests, but Drive publishes no size or digest of its own to corroborate
against, unlike the Hugging Face rows):

```bash
mkdir -p ckpts/C01
python -m pip install gdown
python -m gdown 1YhRPXQQlWmXr5404j_RjiHtvlarHWyPx -O ckpts/C01/SMA-AZO-VGG_DVS128-Gesture.pth
python -m gdown 1zXzSUedGMBkqiCIQNKvLPQxoCaDi-uGQ -O ckpts/C01/SMA-AZO-VGG_CIFAR10-DVS.pth
```

Verify everything against the manifest in one pass:

```bash
sha256sum -c <<'SUMS'
e4449e1a7a3271a52e2be4d3dbb90a3baac13e1cf6394c2e9ed7f0ef68d66dbf  ckpts/A02/cifar100/model_best.pth.tar
13a5e878b0d21590d7091efdb0d3d8b3759ce5134cc3ce83b7117959bf5b662d  ckpts/A04/cifar100/model_best.pth.tar
5a591db6049da28c2356ca56b61ffb6ee5bd5a8fe5ce3d4d2f3b3722a81373b7  ckpts/A07/model.safetensors
8e089eb159099dc848c8ab0c6c741bb746198be3c25dfc58808a3667b8550dc0  ckpts/A07/config.json
d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66  ckpts/A07/tokenizer.json
1f92c93d2e31974acffab86d48e80a9357acb18c68346c5472ef1d12c72d5d77  ckpts/A07/tokenizer_config.json
b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3  ckpts/A07/special_tokens_map.json
07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  ckpts/A07/vocab.txt
97c83b528447b58164e649622576a07886531559348c42ff63b11eac54481f9f  ckpts/B09/vgg16_final_dict.pth.tar
609d7955f4b44d0db5d92f6b2587f72206612e2284c5dc06a544f80c07301a88  ckpts/B09/resnet19_final_dict.pth.tar
8f96a44fcfe4c67ef44f3c0837aa2458e22fec8c7a94b7b7051983698865b492  ckpts/B09/alexnet_final_dict.pth.tar
0701c30e349eed358a2b146e8373431f384674941259ce85491c0949a0076591  ckpts/C01/SMA-AZO-VGG_DVS128-Gesture.pth
1f6dd922f00c898f9ec1111d7cdffb83dcda942a7c6d2418aeafac58dc02dc31  ckpts/C01/SMA-AZO-VGG_CIFAR10-DVS.pth
SUMS
```

Treat the checkpoint files as untrusted data: do not unpickle, extract, or run
them outside the loaders. `benchmarks.py` loads them with `weights_only=True`,
and A07 through `safetensors`.

## 4. Datasets: verify the shared paths

**Nothing needs to be copied.** `/work/hdd` and `/work/nvme` are "Shared
between Delta and DeltaAI" ([data management][data]), so the event data Delta
staged is already readable from DeltaAI at its existing path. Only `/u` differs
between the two systems.

`benchmarks.py` resolves everything below `DATA_ROOT = <repo>/data`
(`benchmarks.py:27`), with four roots:

| Path | Used by | Needed for A02/A04? | How it is populated |
|---|---|---|---|
| `data/hf_cifar` | A02, A04, A08, B09 | **yes** | `datasets` cache, downloaded on first use (`benchmarks.py:202`) |
| `data/hf` | A07 | no | GLUE SST-2 `datasets` cache, downloaded on first use (`benchmarks.py:437`) |
| `data/DVS128Gesture` | C01 | no | already staged on `/work/nvme`, link it |
| `data/cifar10dvs/{train,test}_dataset` | C01 | no | already staged on `/work/nvme`, link it |

For the A02/A04 scope, the only dataset is CIFAR-100, which the `datasets`
library downloads into `data/hf_cifar` by itself on first use (about 260M
covering both CIFAR-10 and CIFAR-100). There is nothing to stage by hand.

The C01 event data is listed for completeness. Verify the shared paths resolve,
then link them exactly as Delta does — do not copy 71G of frames:

```bash
NVME=/work/nvme/bebv/yyu9/neuro_cache_bench_trace/data
ls -d "$NVME/cifar10dvs" "$NVME/DVS128Gesture" || echo "SHARED EVENT DATA NOT VISIBLE"
du -sh "$NVME/cifar10dvs" "$NVME/DVS128Gesture"   # expect ~61G and ~10G

mkdir -p data
ln -sfn "$NVME/cifar10dvs"     data/cifar10dvs
ln -sfn "$NVME/DVS128Gesture"  data/DVS128Gesture
```

If those paths do not resolve, check your project code and `$USER` against the
`accounts` and `quota` commands before assuming the data is gone; the NCSA
convention is `NVME=/work/nvme/<account>/$USER` ([data management][data]).

Two cautions if a future campaign does rebuild the event data rather than
reusing the staged copy. First, the frame parameters are load-bearing:
`frames_number=10 split_by=number` for CIFAR10-DVS and `frames_number=16
split_by=number` for DVS128-Gesture, matching the `T10` and `T16` in the C01
trace directory names — converting with other values silently produces a
different dataset. Second, DVS128-Gesture is not auto-downloadable and
CIFAR10-DVS has no native train/test split, so the train root reuses the test
download through a symlink; `slurm/convert_interactive.sbatch` holds the
conversion with its sample-count assertions, which is the only check that a
conversion finished rather than stopping partway.

If the event files are missing, C01 reports the affected workload as `BLOCKED`
and the other workload still captures, so check the log for `BLOCKED` rather
than assuming a clean exit means two traces.

## 5. Capture commands

Only for the benchmarks the pre-check reported `MISSING`. Both use the same
invariants, `--n 500 --seed 0 --device cuda`, need no chunking, and fit in 64G
of host RAM:

```bash
mkdir -p "$OUT_ROOT"/{A02,A04}

python capture_traces.py --benchmark A02 --n 500 --seed 0 --device cuda --out "$OUT_ROOT/A02"
python capture_traces.py --benchmark A04 --n 500 --seed 0 --device cuda --out "$OUT_ROOT/A04"
```

Each writes one trace directory, named in the table under **Scope** above. Run
them in series.

Then mirror the new directories into the shared location, so Delta and DeltaAI
see the same set. This is **add-only**: `--ignore-existing` and no `--delete`,
because the six Nov 7 traces listed in section 7 already live there and must
not be touched.

```bash
rsync -a --ignore-existing \
  "$OUT_ROOT/A02/spikformer_4_384_cifar100_T4_n500" "$SHARED/A02/"
rsync -a --ignore-existing \
  "$OUT_ROOT/A04/sdt_2_512_cifar100_T4_n500" "$SHARED/A04/"
ls "$SHARED"/A02 "$SHARED"/A04
```

`$SHARED/A02` should then list `spikformer_4_384_cifar100_T4_n500` and
`$SHARED/A04` `sdt_2_512_cifar100_T4_n500` — one directory each, since the
CIFAR-10 names no longer exist (see **Scope**). If a `*_cifar100_*` directory
is already present, the capture has already been mirrored and
`--ignore-existing` correctly left it alone; it is not an error.

Do **not** run the four below; they are listed only so the chunking and memory
figures are on record if a later campaign needs them. The Delta guidance
translates as: `CHUNK=25` becomes `--chunk 25`, and `--mem=160g` becomes "make
sure ~80G of host RAM is actually free", because no scheduler reserves it here.

| Benchmark | `--chunk` | Host RAM to keep free | Trace dirs | Trace size |
|---|---:|---:|---:|---:|
| A07 | 25 | ~80G (measured `MaxRSS` 72.3 GiB) | 1 | 20G |
| A08 | — | 64G is enough | 1 | small |
| B09 | — | 64G is enough | 2 | small |
| C01 | 25 | ~15G (measured `MaxRSS` 14.4 GiB) | 2 | 6G |

Why those two need chunking: the **unchunked A07 capture exceeded 64G of host
RAM**, and the **unchunked C01 capture exhausted CUDA memory**. `--chunk` is
what makes them fit — it splits the 500 samples into forward passes of 25
without changing which samples are used. The 20G/6G figures are output disk
usage, and the `MaxRSS` figures are host RAM, not GPU memory.

Validate what you captured; the validator exits nonzero only when loading a
layer or its parameters fails, and prints `WARN` with a canonical `exclusion_reason()` for
deliberately replay-excluded GEMM or non-3x3 layers:

```bash
find "$OUT_ROOT" -type f -path '*_n500/meta.json' -print0 | sort -z |
  while IFS= read -r -d '' meta; do
    dir="${meta%/meta.json}"
    echo "VALIDATE $dir"
    python validate_trace.py "$dir" || echo "FAIL $dir"
  done
```

## 6. Reproducibility check

Sample selection is a pure function of the seed and the dataset length:
`capture_traces.py:382-383` draws
`np.random.default_rng(seed).choice(len(dataset), n, replace=False)` and sorts
it, and the chunked path only slices that same index list
(`capture_traces.py:238-241`). So a DeltaAI capture is comparable to the Delta one
if and only if it selected the same indices, and every capture records them in
its `meta.json` under `subset` (`capture_traces.py:431-437`: `n`, `seed`,
`sample_indices`, and a `selected_by` string).

`out_t13_cpu_selection/selection_seed0_n500.json` is the committed reference:
the eight expected index lists, keyed by workload, each with the dataset length
it was drawn from.

| Selection key | Dataset length |
|---|---:|
| A02 | 10000 |
| A04 | 10000 |
| A07 | 872 |
| A08 | 10000 |
| B09/vgg16 | 10000 |
| B09/resnet19 | 10000 |
| C01/CIFAR10-DVS | 10000 |
| C01/DVS128-Gesture | 1176 |

Run this after capturing. It is the gate: a mismatch means the dataset or the
split differs from the campaign, and the trace is not interchangeable with the
Delta one even if `validate_trace.py` passes.

```bash
python - <<'PY'
import json, os, sys
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
ref = json.loads(Path("out_t13_cpu_selection/selection_seed0_n500.json").read_text())["benchmarks"]

# trace-dir arch keyword -> selection key; B09 and C01 emit two workloads each.
KEY = {
    "spikformer": "A02",
    "sdt": "A04",
    "spikinglm": "A07",
    "vgg16_cifar10_loas": "A08",
    "loas_vgg16": "B09/vgg16",
    "loas_resnet19": "B09/resnet19",
    "cifar10_dvs": "C01/CIFAR10-DVS",
    "dvs128_gesture": "C01/DVS128-Gesture",
}

failures = 0
for meta_path in sorted(root.glob("*/*_n500/meta.json")):
    meta = json.loads(meta_path.read_text())
    subset = meta["subset"]
    name = meta_path.parent.name
    key = next((k for kw, k in KEY.items() if kw in name), None)
    if key is None:
        print(f"UNMAPPED {meta_path.parent} (add its arch keyword to KEY)")
        failures += 1
        continue
    ok = (subset["seed"] == 0
          and subset["n"] == 500
          and subset["sample_indices"] == ref[key]["sample_indices"])
    print(f"{'MATCH' if ok else 'MISMATCH'} {key} {meta_path.parent}")
    if not ok:
        failures += 1
        got, want = subset["sample_indices"], ref[key]["sample_indices"]
        print(f"  seed={subset['seed']} n={subset['n']} len={len(got)} (want 500)")
        print(f"  first differing: "
              f"{next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), 'n/a')}")

missing = set(ref) - {
    k for p in root.glob("*/*_n500/meta.json")
    for kw, k in KEY.items() if kw in p.parent.name
}
if missing:
    print(f"NOT CAPTURED {sorted(missing)}")

print(f"{'PASS' if not failures and not missing else 'FAIL'} failures={failures}")
sys.exit(1 if failures or missing else 0)
PY
```

`out_t13_cpu_selection/A02_seed0_n500.json` holds the A02 list on its own, in
the same shape minus the `benchmarks` wrapper; it is a convenience copy, not a
second reference.

Expect **one** `MATCH A02` line and one `MATCH A04` line: there is exactly one
n=500 directory per architecture (see **Scope**). A second line per
architecture would appear only if a CIFAR-10 capture were restored beside the
CIFAR-100 one — the keyword map keys on the architecture, which both share, and
both would match, because the CIFAR-10 and CIFAR-100 test splits are each 10000
samples, so seed 0 draws the same indices from both. That would be expected,
not a duplicate-detection failure; it is simply not the state of the tree
today. `NOT CAPTURED` will likewise list nothing
only if the other six Nov 7 trace directories are present under `$OUT_ROOT`; if
they were left on Delta, it lists them and the script exits nonzero. That is
expected — read the per-directory `MATCH`/`MISMATCH` lines, not just the exit
code.

Note on the chunked benchmarks: identical indices do **not** promise
bit-identical tensors. `capture_traces.py:32-36` states that the
chunked-vs-unchunked reference comparison was done only at n=5, chunk=2, and
that those numbers do not bound the n=500 difference, because an unchunked
n=500 reference cannot be produced (it OOMs). A07 and C01 n=500 traces carry a
`numerical_note` in their `meta.json` for this reason.

## 7. Audit verdicts

The scope in the **Scope** section above — A02 and A04 only, `*_cifar100_*`
directories checked for first, new captures mirrored add-only into
`/work/nvme/bebv/yyu9/input_trace_delta/` — is the operative instruction. This
section is the per-trace audit record behind it, copied from
`neuro_cache_bench_trace/out_t16_trace_audit.md`, where each directory was
checked with `validate_trace.py` on the CPU.

| benchmark | trace directory | `captured_at` (UTC) | `validate_trace.py` | verdict |
|---|---|---|---|---|
| A02 | `A02/spikformer_4_384_cifar100_T4_n500` | 2026-11-10 02:19:52 | PASS (exit 0) | **valid** — recaptured after the CIFAR-100 loader/checkpoint fix; all 12 layers passed |
| A04 | `A04/sdt_2_512_cifar100_T4_n500` | 2026-11-10 03:54:16 | PASS (exit 0) | **valid** — recaptured after the CIFAR-100 loader/checkpoint fix; all 16 layers passed |
| A07 | `A07/spikinglm_t4_T4_n500` | 2026-11-07 13:34:01 | PASS (exit 0) | **valid (structural only)** — `chunk_size=25`; all 72 layers passed |
| A08 | `A08/vgg16_cifar10_loas_T32_n500` | 2026-11-07 12:59:56 | PASS (exit 0) | **valid** — loader unchanged; all 13 layers passed |
| B09 | `B09/loas_resnet19_T4_n500` | 2026-11-07 13:01:49 | PASS (exit 0; 6 expected WARNs) | **valid** — the WARNs are the validator's expected replay exclusions for stride-2 layers |
| B09 | `B09/loas_vgg16_T4_n500` | 2026-11-07 13:01:46 | PASS (exit 0) | **valid** — loader unchanged; all 13 layers passed |
| C01 | `C01/sma_azo_vgg_cifar10_dvs_T10_n500` | 2026-11-07 13:40:09 | PASS (exit 0) | **valid (structural only)** — `chunk_size=25`; all 6 layers passed |
| C01 | `C01/sma_azo_vgg_dvs128_gesture_T16_n500` | 2026-11-07 13:39:55 | PASS (exit 0) | **valid (structural only)** — `chunk_size=25`; all 6 layers passed |

Every directory is loader-compatible with the current `benchmarks.py`.

**"Valid (structural only)" is a weaker claim than "valid."** A07 and both C01
workloads were captured with `chunk_size=25`, so their n=500 pass says the
trace is structurally well-formed; it is not numerical-equivalence evidence,
because no unchunked n=500 reference exists to compare against (it OOMs). This
is the same qualification as the closing note of section 6, and the
`numerical_note` in those directories' `meta.json` records it per trace.

On the seed: the campaign command passed `--seed 0`, and
`out_t13_cpu_selection/selection_seed0_n500.json` records the sorted
`numpy.random.default_rng(0)` selections, which match the current A02/A04
selections. The Nov 7 manifests do not independently record the GPU capture
seed, so for the six Nov 7 traces that remains an open provenance gap rather
than a fact closed by inference from matching indices.

If a verdict in that table contradicts the Scope section,
stop and ask rather than capturing: the point of the audit is that nothing
valid gets regenerated.
