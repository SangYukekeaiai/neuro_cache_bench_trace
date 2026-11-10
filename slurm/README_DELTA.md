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

## Interactive per-benchmark capture (n=500, seed 0)

`capture_delta.sbatch` submits all six benchmarks as one array under a single
resource request, so it cannot give A07 more memory than the rest. The
procedure actually used for the n=500 seed-0 set is therefore one
`capture_interactive.sbatch` job per benchmark, each sized for that benchmark.
`capture_interactive.sbatch` takes the benchmark code as its only argument,
reads chunking from the `CHUNK` environment variable, and runs the same
`capture_traces.py --n 500 --seed 0 --device cuda` invocation plus
`validate_trace.py` on every `*_n500/meta.json` it produced.

Submit from the side-project checkout
(`/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace`):

```bash
sbatch --job-name=trace-capture-A02 slurm/capture_interactive.sbatch A02
sbatch --job-name=trace-capture-A04 slurm/capture_interactive.sbatch A04
CHUNK=25 sbatch --mem=160g --job-name=trace-chunk-A07 slurm/capture_interactive.sbatch A07
sbatch --job-name=trace-capture-A08 slurm/capture_interactive.sbatch A08
sbatch --job-name=trace-capture-B09 slurm/capture_interactive.sbatch B09
CHUNK=25 sbatch --job-name=trace-chunk-C01 slurm/capture_interactive.sbatch C01
```

`CHUNK` is exported to the job environment by `sbatch`, so prefixing the
submit line is enough; the script turns it into `--chunk 25`
(`capture_interactive.sbatch:36-37`). `--mem=160g` on the command line
overrides the script's `#SBATCH --mem=64g` (`capture_interactive.sbatch:8`).
A07 and C01 are the only benchmarks that need either: the unchunked A07 capture exceeded
64G host RAM, and unchunked C01 exhausted CUDA memory.

| Benchmark | `CHUNK` | Memory request | Expected trace dirs |
|---|---:|---:|---:|
| A02 | unset | 64G (script default) | 1 |
| A04 | unset | 64G (script default) | 1 |
| A07 | 25 | 160G (`--mem=160g`) | 1 |
| A08 | unset | 64G (script default) | 1 |
| B09 | unset | 64G (script default) | 2 |
| C01 | 25 | 64G (script default) | 2 |

B09 and C01 each emit two trace directories; the script encodes that
expectation itself and fails the job if the count does not match
(`capture_interactive.sbatch:54-55`, checked at `:57-60`). Each job writes
`data/capture-%x-%j.log` and ends with a single `SUMMARY` line giving
`capture_exit`, `validation_exit`, `traces`, `expected`, and timings, followed
by `du -sh` of the output directory, so the log tail is the pass/fail record.

Outputs land in `$WCACHE/examples/input_trace/delta/$B/`, which the wcache
checkout ignores via `.gitignore:52`.

### Jobs actually run

Every job's `SUMMARY` line and its `sacct` batch-step `MaxRSS`:

| Benchmark | Job | `--chunk` | `--mem` | Result | MaxRSS | GiB | Elapsed |
|---|---:|---:|---:|---|---:|---:|---:|
| A02 | 23638403 | — | 64G | PASS, 1/1 traces (superseded) | 9988652K | 9.5 | 00:01:06 |
| A04 | 23638434 | — | 64G | PASS, 1/1 traces (superseded) | 11929260K | 11.4 | 00:00:49 |
| A07 | 23638454 | — | 64G | FAIL, `capture_exit=137`, 0/1 traces | 67074792K | 63.9 | 00:01:16 |
| A07 | 23638908 | 25 | 160G | PASS, 1/1 traces | 75765692K | 72.3 | 00:07:43 |
| A08 | 23638493 | — | 64G | PASS, 1/1 traces | 18280884K | 17.4 | 00:00:46 |
| B09 | 23638557 | — | 64G | PASS, 2/2 traces | 7250964K | 6.9 | 00:00:39 |
| C01 | 23638617 | — | 64G | FAIL, `capture_exit=1`, 0/2 traces | 2866252K | 2.7 | 00:00:21 |
| C01 | 23639155 | 25 | 64G | PASS, 2/2 traces | 15142132K | 14.4 | 00:01:23 |

The two FAIL rows are why the chunking exists, and they fail differently.
A07's `capture_exit=137` is SIGKILL: the unchunked capture was killed for
exceeding the 64G host-memory request. `sacct` puts job 23638454 in state
`OUT_OF_MEMORY` with a batch-step `MaxRSS` of 67074792K — 63.9 GiB, sitting
right at the 64G cap at the moment the kernel killed it, which is the direct
measurement behind the claim rather than the exit code alone. C01's `capture_exit=1` is a Python-level CUDA out-of-memory from the
unchunked forward pass, which is why it reports a small 2.7 GiB host peak — the
memory it ran out of was on the GPU. Chunked A07 then peaked at 72.3 GiB of
host RAM, above the 64G default, so it needs both `CHUNK=25` and `--mem=160g`;
chunked C01 peaked at 14.4 GiB and needs only `CHUNK=25`.

A07 is also the one slow capture: 417 capture seconds against 16-55 for the
unchunked benchmarks, the cost of 20 forward passes of 25 samples instead of
one.

The two rows marked *superseded* are the Nov 7 CIFAR-10 captures. They produced
`A02/spikformer_4_384_T4_n500` and `A04/sdt_2_512_T4_n500`, directories that no
longer exist: commit `fbae5bb` switched A02/A04 to the CIFAR-100 loaders and
checkpoints, which tag the architecture differently, and the captures were
remade under the new names. Their memory and timing figures still stand as
measurements of this Slurm path; their outputs are not part of the current
trace set.

One further job was submitted and produced nothing: `23689507`
(`trace-capture-A02-lean`, 32G) was meant to re-capture A02 and then A04 under
the CIFAR-100 loaders, but `sacct` reports it `FAILED` after 00:00:05 with a
batch-step `MaxRSS` of 1116K, and `data/capture-A02-23689507.log` holds a
single line — `slurm/capture_interactive.sbatch: Permission denied`. It never
reached `capture_traces.py`, so it has no result row and no output. The
CIFAR-100 captures that exist were made interactively on a GH200 node instead;
see the next subsection.

### The current A02/A04 captures (GH200, interactive, no Slurm)

The A02 and A04 traces in use today were **not** produced by any job in the
table above. Their `meta.json` records a capture on a GH200 node after the
CIFAR-100 switch — on these systems that means DeltaAI: `sinfo` shows Delta's
GPU partitions carrying A40, A100, H200 and MI100/MI210, and **no GH200 at
all** (`sinfo -o "%P %G"` returns zero matches for `gh200`), while DeltaAI's
nodes are 4-way GH200 Grace Hopper superchips
([`../DTAI_HANDOFF.md`](../DTAI_HANDOFF.md) section 2). Note Delta's
`gpuH200x8` partition: H200 is a discrete Hopper GPU, not the GH200 Grace
Hopper superchip, and the near-identical names are the one way to misread this.
That chain is an inference from the device string below, not something the file
states:

| Trace dir | `captured_at` (UTC) | device | dataset | `n` / seed | `chunk_size` |
|---|---|---|---|---:|---:|
| `A02/spikformer_4_384_cifar100_T4_n500` | 2026-11-10 02:19:52 | `cuda:0 (NVIDIA GH200 120GB)` | `cifar100/test` | 500 / 0 | none |
| `A04/sdt_2_512_cifar100_T4_n500` | 2026-11-10 03:54:16 | `cuda:0 (NVIDIA GH200 120GB)` | `cifar100/test` | 500 / 0 | none |

Both pass `validate_trace.py` (12 and 16 layers; see
`../out_t16_trace_audit.md`) and both are mirrored into
`/work/nvme/bebv/yyu9/input_trace_delta/`. Neither needed chunking: the Nov 7
Slurm runs of the same two benchmarks peaked at 9.5 and 11.4 GiB, far below the
host-memory limit.

What is **not** recorded: an interactive session leaves no job log, and
`meta.json` does not store the command line or the host name, so the exact
invocation behind these two directories cannot be recovered from the artifacts.
Everything that bears on reproducibility is recorded — seed, `n`, dataset, the
full index list — and is checked below. The step-by-step procedure for
repeating this capture on a plain GPU server is
[`../DTAI_HANDOFF.md`](../DTAI_HANDOFF.md).

### Seed-0 reproducibility

Sample selection is a pure function of `--seed` and the dataset length:
`capture_traces.py:382-383` builds `np.random.default_rng(args.seed)` and takes
`rng.choice(len(selected_dataset), args.n, replace=False)`, sorted. Nothing in
the chunked path changes which samples are chosen — `--chunk` only slices that
same index list into forward passes (`capture_traces.py:238-241`).

Every capture records the selection in its `meta.json` under `subset`
(`capture_traces.py:431-437`): `n`, `seed`, the full `sample_indices` list, and
a `selected_by` string naming the selection expression.
The run is reproducible by re-reading that field rather than by trusting the
command line.

The reference to read it against is
[`../out_t13_cpu_selection/selection_seed0_n500.json`](../out_t13_cpu_selection/selection_seed0_n500.json),
which holds the full expected 500-index list for all eight workloads, computed
on the CPU from the seed and the dataset length alone. **Every one of the eight
captured directories matches that reference across all 500 indices**, not
merely on a prefix — including the two CIFAR-100 recaptures, which were made on
different hardware by a different route than the other six. That full-list
agreement is the reproducibility evidence; the prefix column below is only a
reading aid. Observed across the n=500 set, all with `subset.seed = 0`:

| Trace dir | `chunk_size` | full 500-index match | first five `sample_indices` |
|---|---:|---|---|
| A02/spikformer_4_384_cifar100_T4_n500 | — | yes | 26, 51, 63, 79, 97 |
| A04/sdt_2_512_cifar100_T4_n500 | — | yes | 26, 51, 63, 79, 97 |
| A07/spikinglm_t4_T4_n500 | 25 | yes | 1, 2, 3, 4, 5 |
| A08/vgg16_cifar10_loas_T32_n500 | — | yes | 26, 51, 63, 79, 97 |
| B09/loas_resnet19_T4_n500 | — | yes | 26, 51, 63, 79, 97 |
| B09/loas_vgg16_T4_n500 | — | yes | 26, 51, 63, 79, 97 |
| C01/sma_azo_vgg_cifar10_dvs_T10_n500 | 25 | yes | 26, 51, 63, 79, 97 |
| C01/sma_azo_vgg_dvs128_gesture_T16_n500 | 25 | yes | 1, 3, 5, 6, 8 |

The prefix is shared by every benchmark whose evaluation split has the same
length, which is the expected consequence of seeding on the dataset length
alone: those six have `max(sample_indices) = 9987`, while A07 tops out at 869
and C01/DVS128-Gesture at 1174, so their draws differ.

Numerical caveat for the chunked runs: `capture_traces.py:32-36` states that
the reference chunk-vs-unchunked comparison was done only at n=5, chunk=2, and
that those numbers do not bound the n=500 difference, because an unchunked
n=500 reference cannot be produced (OOM). A07 and C01 n=500 traces therefore
carry `numerical_note` in their `meta.json` and are not bit-identical
references.
