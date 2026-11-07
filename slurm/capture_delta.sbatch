#!/usr/bin/env bash
#SBATCH --account=bebv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

PROJECT_ROOT="/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace"
WCACHE="/u/yyu9/projects/neuro_cache_wcache"
SUITE="delta"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

BENCHMARKS=(A02 A04 A07 A08 B09 C01)
B="${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT_ROOT"

/u/yyu9/miniconda3/bin/conda run --no-capture-output -n base "$PYTHON_BIN" capture_traces.py \
  --benchmark "$B" \
  --n 500 \
  --seed 0 \
  --out "$WCACHE/examples/input_trace/$SUITE/$B" \
  --device cuda
