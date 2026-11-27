#!/usr/bin/env bash
# Full pipeline on a Mac CPU: data -> 11 training runs -> evaluation -> benchmark/INT8.
# Expected ~7 h on an M2 Pro; every run has a time cap so the worst case stays under 12 h.
# Usage:  ./run_all.sh            (keeps the Mac awake with caffeinate; safe to re-run, finished runs are skipped)
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${LU_AWAKE:-}" ]] && command -v caffeinate >/dev/null; then
  LU_AWAKE=1 exec caffeinate -i "$0" "$@"
fi
PY=.venv/bin/python
mkdir -p results
LOG=results/run_all.log
start=$(date +%s)
echo "== start $(date)" | tee -a "$LOG"
$PY generate_data.py 2>&1 | tee -a "$LOG"
for cfg in configs/*.json; do
  name=$(basename "$cfg" .json)
  if [[ -f "runs/$name/model.pt" ]]; then echo "skip $name (already trained)" | tee -a "$LOG"; continue; fi
  $PY train.py "$cfg" 2>&1 | tee -a "$LOG"
done
$PY evaluate.py 2>&1 | tee -a "$LOG"
$PY benchmark.py --int8 e1_liteunet e5_w8 2>&1 | tee -a "$LOG"
echo "== done in $(( ($(date +%s) - start) / 60 )) min  ->  results/summary.md" | tee -a "$LOG"
