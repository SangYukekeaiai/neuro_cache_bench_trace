# Benchmark trace capture

This side project captures input spike traces in the format consumed by
`neuro_cache-wcache`.  The six registry entries load the pinned upstream model
definitions and checkpoints from `upstream/` and `ckpts/`; `synthetic` remains
the deterministic local smoke benchmark.

## Environment

The checked-out environment used for the local GPU trials contains:

```text
torch==2.15.0+cu132       torchvision==0.29.2+cu132
numpy==2.5.4              spikingjelly==0.0.0.0.14
timm==0.6.13              braincog==git f9b879f75da2247a9f0c31864a947f0e5d2f3dab
transformers==4.57.0      datasets==3.3.2
safetensors==0.8.0        accelerate==1.15.0
huggingface_hub==0.36.2
```

For a new CUDA machine, create the base environment first, then install the
benchmark dependencies (the pinned versions above are the known-good set):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision numpy
.venv/bin/python -m pip install \
  spikingjelly==0.0.0.0.14 timm==0.6.13 transformers==4.57.0 \
  datasets==3.3.2 safetensors==0.8.0 accelerate==1.15.0
.venv/bin/python -m pip install git+https://github.com/BrainCog-X/Brain-Cog.git@f9b879f75da2247a9f0c31864a947f0e5d2f3dab
```

`benchmarks.py` imports model definitions lazily from the pristine upstream
clones.  Checkpoints are loaded with `weights_only=True`; A07 uses
`safetensors`.

SpikingJelly 0.0.0.0.14 does not work with NumPy 2 when it converts CIFAR10-DVS
events: without the fix, `events_np/` stays empty and loading fails with
"FileNotFoundError: Found no valid file for the classes".  Change line 62 of
`datasets/cifar10_dvs.py` from `np.fromstring(...)` to
`data = np.frombuffer(data, dtype='>u4')`:

```bash
sed -i "s/np.fromstring(data, dtype='>u4')/np.frombuffer(data, dtype='>u4')/" \
  .venv/lib/python3.12/site-packages/spikingjelly/datasets/cifar10_dvs.py
```

## Capture

Each invocation uses one sorted NumPy-selected subset and one forward pass.
For a registry entry that contains multiple workloads (B09 and C01), each
workload gets its own `<arch>_T<T>_n<N>` directory below the requested output.

```bash
for id in A02 A04 A07 A08 B09 C01; do
  .venv/bin/python capture_traces.py \
    --benchmark "$id" --n 5 --seed 0 --out "./out_real/$id" --device cuda
done
```

Individual commands are:

```bash
.venv/bin/python capture_traces.py --benchmark A02 --n 5 --seed 0 --out ./out_real/A02 --device cuda
.venv/bin/python capture_traces.py --benchmark A04 --n 5 --seed 0 --out ./out_real/A04 --device cuda
.venv/bin/python capture_traces.py --benchmark A07 --n 5 --seed 0 --out ./out_real/A07 --device cuda
.venv/bin/python capture_traces.py --benchmark A08 --n 5 --seed 0 --out ./out_real/A08 --device cuda
.venv/bin/python capture_traces.py --benchmark B09 --n 5 --seed 0 --out ./out_real/B09 --device cuda
.venv/bin/python capture_traces.py --benchmark C01 --n 5 --seed 0 --out ./out_real/C01 --device cuda
```

The preferred test/validation split is selected when it has at least `N`
samples; otherwise the loader uses its train split and records that split in
`meta.json`.  A weight-bearing layer whose input is not exactly 0/1 is not
captured: it is recorded in `excluded_layers` with a reason.  Captured layers
retain the hard binary assertion.  Attention q@k products have no weight
module and are not targets.  A declared target that does not fire is a hard
error; use `--allow-unfired` only for a later benchmark-specific exception.

```bash
.venv/bin/python capture_traces.py \
  --benchmark synthetic --n 5 --seed 0 --out ./out --device cuda
.venv/bin/python capture_traces.py \
  --benchmark synthetic --n 5 --seed 0 --out ./out --device cuda --allow-unfired
```

C01's event datasets are not bundled.  If SpikingJelly cannot find the
manually fetched event files, the affected workload is reported as `BLOCKED`
and the other workloads continue; place the DVS128-Gesture files under
`data/DVS128Gesture/` and rerun the command.  SpikingJelly downloads
CIFAR10-DVS itself, but both roots must exist first and neither `download/`
subdirectory may exist in advance.  CIFAR10-DVS has no native train/test
split, so after the test root's download finishes, the train root can reuse
it through a symlink:

```bash
mkdir -p data/cifar10dvs/train_dataset data/cifar10dvs/test_dataset
# after test_dataset/download is complete:
ln -s ../test_dataset/download data/cifar10dvs/train_dataset/download
```

## Validate

`validate_trace.py` adds the canonical loader source automatically.  Its
default is `/home/ya867177/neuro_cache-wcache/src`; override it with
`WCACHE_SRC` when the checkout is elsewhere.  It imports the loader from the
`lays` package (`lays.weight_trace_gen.trace`):

```bash
.venv/bin/python validate_trace.py \
  /home/ya867177/neuro_cache-wcache/input_trace/loas/vgg16_T4_n5
.venv/bin/python validate_trace.py out/synthetic_spiking_T4_n5
.venv/bin/python validate_trace.py out_real/A02/spikformer_4_384_T4_n5
WCACHE_SRC=/path/to/neuro_cache-wcache/src \
  .venv/bin/python validate_trace.py out/synthetic_spiking_T4_n5
```

Run the capture-hook regression test alongside these capture and validation commands:
`.venv/bin/python test_capture_traces.py`

The validator prints `PASS` for replayable layers and `WARN` with the
canonical `exclusion_reason()` for intentionally replay-excluded GEMM or
non-3x3 layers.  It returns nonzero only when loading a layer or its required
parameters fails.

## Sanity report

`out_real/TRIALS.md` records the 64-sample seeded evaluation, captured and
excluded layer counts/reasons, trace directory size, and validator exit code
for each workload.  A07 is a pretrained base without an SST-2 head, so its
entry reports masked-LM pseudo-loss against a random-init baseline instead of
classification accuracy.

The report's evaluation rows are produced by the benchmark module itself.  The
command below selects 64 samples from each test/validation split with NumPy's
`default_rng(0)`, evaluates each workload in one forward pass, and uses the
same 15%-masked token batch for A07 and its random-init baseline:

```bash
.venv/bin/python benchmarks.py --evaluate \
  --benchmark A02 --benchmark A04 --benchmark A07 --benchmark A08 --benchmark B09 \
  --device cuda --n 64 --seed 0
```

The command prints the accuracy or pseudo-loss values copied into
`out_real/TRIALS.md`.  C01 is evaluated separately on the CPU, because a
64-sample T=16 batch does not fit next to other work on the shared GPU:

```bash
.venv/bin/python benchmarks.py --evaluate --benchmark C01 --device cpu --n 64 --seed 0
```

## Delta

The site-specific array script and submit/check instructions are in
[slurm/README_DELTA.md](slurm/README_DELTA.md).
