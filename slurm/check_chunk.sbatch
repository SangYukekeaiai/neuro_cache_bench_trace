#!/usr/bin/env bash
#SBATCH --job-name=trace-check-chunk
#SBATCH --account=bebv-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --chdir=/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace
#SBATCH --output=data/check-chunk-%j.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
B="${1:-C01}"
case "$B" in A07|C01) ;; *) exit 2 ;; esac
PROJECT_ROOT="/u/yyu9/projects/neuro_cache_wcache/neuro_cache_bench_trace"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
CONDA_BIN="/u/yyu9/miniconda3/bin/conda"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export WCACHE_SRC="/u/yyu9/projects/neuro_cache_wcache/src"
cd "$PROJECT_ROOT"

setsid "$CONDA_BIN" run --no-capture-output -n base "$PYTHON_BIN" capture_traces.py \
  --benchmark "$B" --n 5 --seed 0 --chunk 2 --out "out_chunk_check/$B" --device cuda
for trace in out_chunk_check/"$B"/*_n5; do
  setsid "$CONDA_BIN" run --no-capture-output -n base "$PYTHON_BIN" validate_trace.py "$trace"
done
"$CONDA_BIN" run --no-capture-output -n base "$PYTHON_BIN" - "$B" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

original = Path("out_smoke_i") / sys.argv[1]
chunked = Path("out_chunk_check") / sys.argv[1]
old_files = {path.relative_to(original) for path in original.glob("*/*.npy")}
new_files = {path.relative_to(chunked) for path in chunked.glob("*/*.npy")}
assert old_files and old_files == new_files, "trace filenames differ"
for relative in sorted(old_files):
    with (original / relative).open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    with (chunked / relative).open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    assert before == after, f"trace checksum mismatch: {relative}: {before} != {after}"
    print(f"SHA256 MATCH {relative} {after}")
for path in sorted(original.glob("*/meta.json")):
    before = json.loads(path.read_text())
    after = json.loads((chunked / path.relative_to(original)).read_text())
    for metadata in (before, after):
        metadata.pop("captured_at")
        metadata["device"].pop("actual")
    assert before == after, f"metadata contract differs: {path.parent.name}"
    print(f"METADATA MATCH {path.parent.name} (excluding timestamp and device identity)")
print(f"EQUIVALENCE PASS files={len(old_files)} chunk=2 n=5")
PY
