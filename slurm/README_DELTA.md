# Delta capture array

`capture_delta.sbatch` is a six-task Slurm array. Its task IDs map to the
benchmark table in this order:

| Array task | Benchmark |
|---:|---|
| 0 | A02 |
| 1 | A04 |
| 2 | A07 |
| 3 | A08 |
| 4 | B09 |
| 5 | C01 |

Each task invokes `capture_traces.py` with 500 samples, seed `0`, and CUDA,
writing under `$WCACHE/input_trace/$SUITE/$B`. The benchmark-specific `$B`
component prevents array tasks from colliding when two models have the same
architecture/timestep/sample-count directory. The six benchmark loaders are
implemented in `benchmarks.py` and selected from the registry. The C01
DVS128-Gesture capture remains blocked until its event files are downloaded on
the target environment; CIFAR10-DVS is downloadable through SpikingJelly, but
the parent directory `data/cifar10dvs/test_dataset` must be created first
because its downloader calls `os.mkdir(download_root)` and `benchmarks.py` does
not currently create that parent.

Before submitting on Delta, verify the `WCACHE`, `SUITE`, and `PYTHON_BIN`
values and the TODO-marked `PROJECT_ROOT` at the top of the script. In
particular, the wcache checkout is under `/u/yyu9` on Delta. Also verify the
account `bebv-delta-gpu` and partition `gpuA40x4` before submitting; these are
site-specific scheduler settings.

Submit the array from the side-project checkout with:

```bash
sbatch --array=0-5 slurm/capture_delta.sbatch
```

The resulting `*_n500` capture directories are ignored by the wcache checkout's
`.gitignore` rule `input_trace/delta/` (currently line 48). Proof from the
wcache checkout:

```text
$ git check-ignore -v input_trace/delta/A02/x_T4_n500/layer_01.npy
.gitignore:48:input_trace/delta/ input_trace/delta/A02/x_T4_n500/layer_01.npy
```

This script cannot be executed on the current machine because it is not a
Delta Slurm environment. The local syntax check is:

```bash
bash -n slurm/capture_delta.sbatch
```
