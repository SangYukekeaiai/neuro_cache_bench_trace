#!/usr/bin/env bash
# TODO(Delta): verify the account before submitting at this site.
#SBATCH --account=bebv-delta-gpu
# TODO(Delta): verify the partition before submitting at this site.
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

# TODO(Delta): verify the side-project checkout path before submitting.
PROJECT_ROOT="/u/yyu9/neuro_cache_bench_trace"
# TODO(Delta): verify this path before submitting; Delta's wcache checkout is
# under /u/yyu9, unlike the checkout on the development machine.
WCACHE="/u/yyu9/neuro_cache-wcache"
# TODO(Delta): verify the suite directory name before submitting.
SUITE="delta"
# TODO(Delta): verify the side-project venv path before submitting.
PYTHON_BIN="/u/yyu9/neuro_cache_bench_trace/.venv/bin/python"

BENCHMARKS=(A02 A04 A07 A08 B09 C01)
B="${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" capture_traces.py \
  --benchmark "$B" \
  --n 500 \
  --seed 0 \
  --out "$WCACHE/input_trace/$SUITE/$B" \
  --device cuda
