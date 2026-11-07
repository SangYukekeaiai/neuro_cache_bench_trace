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
writing under `$WCACHE/examples/input_trace/$SUITE/$B`. The benchmark-specific `$B`
component prevents array tasks from colliding when two models have the same
architecture/timestep/sample-count directory. The six benchmark loaders are
implemented in `benchmarks.py` and selected from the registry. The C01
event datasets and frame conversions are available under `data/`, with the
event-data directories symlinked to `/work/nvme/bebv/yyu9/neuro_cache_bench_trace/data/`.

For n=500, A07 and C01 require `--chunk 25` to fit; use `--mem=160g` for A07.
Measured peak host memory from `sacct` (batch-step `MaxRSS`):

| Benchmark | Completed job | MaxRSS | Approx. GiB | Requested memory | Output size |
|---|---:|---:|---:|---:|---:|
| A07 | 23638908 | 75765692K | 72.3 | 160G | 20G |
| C01 | 23639155 | 15142132K | 14.4 | 64G | 6G |

The 20G/6G values are disk usage, not peak memory. MaxRSS measures host RAM,
not GPU memory. The unchunked A07 capture exceeded 64G host RAM; unchunked
C01 exhausted CUDA memory. For future interactive captures, export `CHUNK=25`
when submitting `slurm/capture_interactive.sbatch`, and override A07's memory
request with `sbatch --mem=160g`. `capture_delta.sbatch` does not set chunking;
its default 64G, unchunked configuration is insufficient for A07/C01 at n=500.

The script uses `PROJECT_ROOT=/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace`
and `WCACHE=/u/yyu9/projects/neuro_cache_wcache`, with `SUITE=delta` and
`PYTHON_BIN=$PROJECT_ROOT/.venv/bin/python`. The account `bebv-delta-gpu`
and partition `gpuA40x4` are verified for this machine. Python runs through
`/u/yyu9/miniconda3/bin/conda run --no-capture-output -n base`; both
`OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` use the allocated CPU count
(16 by default).

Submit the array from the side-project checkout with:

```bash
sbatch --array=0-5 slurm/capture_delta.sbatch
```

The resulting `*_n500` capture directories are ignored by the wcache checkout's
`.gitignore` rule `examples/input_trace/delta/` (line 52). Proof from the
wcache checkout:

```text
$ git check-ignore -v examples/input_trace/delta/A02/x_T4_n500/layer_01.npy
.gitignore:52:examples/input_trace/delta/ examples/input_trace/delta/A02/x_T4_n500/layer_01.npy
```

Check the script's shell syntax before submission:

```bash
bash -n slurm/capture_delta.sbatch
```
