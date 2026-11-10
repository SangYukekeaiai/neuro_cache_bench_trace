#!/usr/bin/env bash
#SBATCH --job-name=trace-dtai
#SBATCH --account=bebv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --chdir=/u/yyu9/neuro_cache_bench_trace
#SBATCH --output=out_dtai/slurm-%x-%j.out

set -euo pipefail

B="${1:-}"
case "$B" in
    A02|A04) ;;
    *) echo "Usage: sbatch $0 {A02|A04}" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

PROJECT_ROOT="/u/yyu9/neuro_cache_bench_trace"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
OUT="./out_dtai/$B"
export WCACHE_SRC="/u/yyu9/neuro_cache_wcache/src"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
mkdir -p "$OUT"
printf 'START %s job=%s time=%s\n' "$B" "$SLURM_JOB_ID" "$(date -Is)"
setsid "$PYTHON_BIN" capture_traces.py \
    --benchmark "$B" \
    --n 500 \
    --seed 0 \
    --out "$OUT" \
    --device cuda
printf 'DONE %s job=%s time=%s\n' "$B" "$SLURM_JOB_ID" "$(date -Is)"
