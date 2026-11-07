#!/usr/bin/env bash
#SBATCH --job-name=trace-smoke
#SBATCH --account=bebv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=4:00:00
#SBATCH --chdir=/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace
#SBATCH --output=out_smoke/slurm-%j.out

set -uo pipefail

PROJECT_ROOT="/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export WCACHE_SRC="/u/yyu9/projects/neuro_cache_wcache/src"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT" || exit 1

BENCHMARKS=(A02 A04 A07 A08 B09 C01)
declare -A capture_rc capture_seconds validation_rc trace_count

for B in "${BENCHMARKS[@]}"; do
    mkdir -p "out_smoke/$B" || exit 1
    if [[ "$B" == C01 ]]; then
        if [[ -d data/cifar10dvs/test_dataset/download &&
              ! -e data/cifar10dvs/train_dataset/download &&
              ! -L data/cifar10dvs/train_dataset/download ]]; then
            ln -s ../test_dataset/download data/cifar10dvs/train_dataset/download || exit 1
        fi
    fi
    printf 'CAPTURE %s\n' "$B"
    capture_start=$SECONDS
    setsid /u/yyu9/miniconda3/bin/conda run --no-capture-output -n base "$PYTHON_BIN" capture_traces.py \
        --benchmark "$B" --n 5 --seed 0 --out "./out_smoke/$B" --device cuda \
        2>&1 | tee "out_smoke/$B/capture.log"
    capture_rc[$B]=${PIPESTATUS[0]}
    capture_seconds[$B]=$((SECONDS - capture_start))
done

setsid /u/yyu9/miniconda3/bin/conda run --no-capture-output -n base "$PYTHON_BIN" test_capture_traces.py \
    2>&1 | tee out_smoke/regression.log
regression_rc=${PIPESTATUS[0]}

for B in "${BENCHMARKS[@]}"; do
    validation_rc[$B]=0
    trace_count[$B]=0
    while IFS= read -r -d '' metadata; do
        trace_dir="${metadata%/meta.json}"
        trace_count[$B]=$(( ${trace_count[$B]} + 1 ))
        printf 'VALIDATE %s\n' "$trace_dir"
        setsid /u/yyu9/miniconda3/bin/conda run --no-capture-output -n base "$PYTHON_BIN" validate_trace.py "$trace_dir" \
            2>&1 | tee "$trace_dir/validation.log"
        result=${PIPESTATUS[0]}
        if (( result != 0 )); then
            validation_rc[$B]=$result
        fi
    done < <(find "out_smoke/$B" -type f -name meta.json -print0 | sort -z)
done

failed=0
if (( regression_rc == 0 )); then
    printf 'REGRESSION PASS exit=0\n'
else
    printf 'REGRESSION FAIL exit=%s\n' "$regression_rc"
    failed=1
fi

for B in "${BENCHMARKS[@]}"; do
    status=PASS
    expected=1
    [[ "$B" == B09 || "$B" == C01 ]] && expected=2
    if (( ${capture_rc[$B]} != 0 || ${validation_rc[$B]} != 0 )); then
        status=FAIL
    elif (( ${trace_count[$B]} < expected )); then
        status=FAIL
        if grep -q "^BLOCKED $B:" "out_smoke/$B/capture.log"; then
            status=BLOCKED
        fi
    fi
    if [[ "$B" == C01 && "$status" == PASS ]] &&
       [[ ! -f out_smoke/C01/sma_azo_vgg_cifar10_dvs_T10_n5/meta.json ||
          ! -f out_smoke/C01/sma_azo_vgg_dvs128_gesture_T16_n5/meta.json ]]; then
        status=FAIL
    fi
    [[ "$status" != PASS ]] && failed=1
    printf 'SUMMARY %s %s capture_exit=%s validation_exit=%s traces=%s capture_seconds=%s\n' \
        "$B" "$status" "${capture_rc[$B]}" "${validation_rc[$B]}" "${trace_count[$B]}" "${capture_seconds[$B]}"
    du -sh "out_smoke/$B"
    grep "^BLOCKED $B:" "out_smoke/$B/capture.log" || true
done

if [[ -d data/cifar10dvs/test_dataset/download &&
      ! -e data/cifar10dvs/train_dataset/download ]]; then
    printf 'C01 RETRY: test download now exists; rerun to link it for the train split.\n'
fi
exit "$failed"
