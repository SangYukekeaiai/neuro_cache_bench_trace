#!/usr/bin/env bash
#SBATCH --account=bebv-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --chdir=/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace
#SBATCH --output=data/capture-%x-%j.log

set -uo pipefail

B="${1:-}"
case "$B" in
  A02|A04|A07|A08|B09|C01) ;;
  *) echo "Usage: sbatch $0 {A02|A04|A07|A08|B09|C01}" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

PROJECT_ROOT="/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace"
WCACHE="/u/yyu9/projects/neuro_cache_wcache"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
CONDA_BIN="/u/yyu9/miniconda3/bin/conda"
OUT="$WCACHE/examples/input_trace/delta/$B"
export WCACHE_SRC="$WCACHE/src"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT" || exit 1
mkdir -p "$OUT" || exit 1

printf 'START %s job=%s time=%s\n' "$B" "$SLURM_JOB_ID" "$(date -Is)"
capture_start=$SECONDS
chunk_args=()
if [[ -n "${CHUNK:-}" ]]; then chunk_args=(--chunk "$CHUNK"); fi
setsid "$CONDA_BIN" run --no-capture-output -n base "$PYTHON_BIN" capture_traces.py \
  --benchmark "$B" --n 500 --seed 0 --out "$OUT" --device cuda "${chunk_args[@]}"
capture_rc=$?
capture_seconds=$((SECONDS - capture_start))

validation_rc=0
trace_count=0
while IFS= read -r -d '' metadata; do
  trace_count=$((trace_count + 1))
  trace_dir="${metadata%/meta.json}"
  printf 'VALIDATE %s\n' "$trace_dir"
  setsid "$CONDA_BIN" run --no-capture-output -n base "$PYTHON_BIN" validate_trace.py "$trace_dir"
  result=$?
  if (( result != 0 )); then validation_rc=$result; fi
done < <(find "$OUT" -type f -path '*_n500/meta.json' -print0 | sort -z)

expected=1
[[ "$B" == B09 || "$B" == C01 ]] && expected=2
status=PASS
failed=0
if (( capture_rc != 0 || validation_rc != 0 || trace_count != expected )); then
  status=FAIL
  failed=1
fi
printf 'SUMMARY %s %s capture_exit=%s validation_exit=%s traces=%s expected=%s capture_seconds=%s total_seconds=%s\n' \
  "$B" "$status" "$capture_rc" "$validation_rc" "$trace_count" "$expected" "$capture_seconds" "$SECONDS"
du -sh "$OUT"
exit "$failed"
