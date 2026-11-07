#!/usr/bin/env bash
#SBATCH --account=bebv-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --chdir=/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace
#SBATCH --output=data/convert-%x-%j.log

set -euo pipefail

STAGE="${1:-${STAGE:-}}"
case "$STAGE" in
  cifar10dvs_test|cifar10dvs_train|dvs128) ;;
  *) echo "Usage: sbatch $0 {cifar10dvs_test|cifar10dvs_train|dvs128}" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

PROJECT_ROOT="/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
cd "$PROJECT_ROOT"

/u/yyu9/miniconda3/bin/conda run --no-capture-output -n base "$PYTHON_BIN" -u - "$STAGE" <<'PY'
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

started = time.monotonic()
stage = sys.argv[1]
print(f"START stage={stage} job={os.environ['SLURM_JOB_ID']} time={datetime.now(timezone.utc).isoformat()}")

from spikingjelly import configure
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

print(f"preprocess_threads={configure.max_threads_number_for_datasets_preprocess} "
      f"OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']} "
      f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

if stage == "dvs128":
    splits = (("train", True, 1176), ("test", False, 288))
else:
    splits = ((stage.removeprefix("cifar10dvs_"), None, 10000),)

for split, train, expected in splits:
    split_started = time.monotonic()
    if stage == "dvs128":
        dataset = DVS128Gesture("data/DVS128Gesture", train=train,
                                data_type="frame", split_by="number", frames_number=16)
    else:
        dataset = CIFAR10DVS(f"data/cifar10dvs/{split}_dataset",
                             data_type="frame", split_by="number", frames_number=10)
    count = len(dataset)
    class_counts = dict(sorted(Counter(label for _, label in dataset.samples).items()))
    print(f"COUNT stage={stage} split={split} samples={count} expected={expected} "
          f"classes={class_counts} elapsed_seconds={time.monotonic() - split_started:.2f}")
    if count != expected:
        raise RuntimeError(f"Incomplete conversion: {stage}/{split}: {count} != {expected}")
    if stage != "dvs128" and (len(class_counts) != 10 or set(class_counts.values()) != {1000}):
        raise RuntimeError(f"Incomplete CIFAR10-DVS classes: {class_counts}")

print(f"COMPLETE stage={stage} elapsed_seconds={time.monotonic() - started:.2f} "
      f"time={datetime.now(timezone.utc).isoformat()}")
PY
