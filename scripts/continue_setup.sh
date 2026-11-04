#!/usr/bin/env bash
# Durable continuation of the user-authorized environment and first baseline.
set -euo pipefail
scripts=$(dirname "$(realpath "$0")")
source "$scripts/common.sh"
: "${SETUP_JOB_DIR:?Missing per-job output directory}"
export SETUP_PHASE=starting
record_state() {
  export SETUP_EXIT_CODE="$?"
  /usr/bin/python3 -B - <<'STATE_PY'
import json,os
from pathlib import Path
from datetime import datetime,timezone
p=Path(os.environ['SETUP_JOB_DIR'])/'state.json'
p.write_text(json.dumps({'phase':os.environ['SETUP_PHASE'], 'exit_code':int(os.environ['SETUP_EXIT_CODE']), 'updated_utc':datetime.now(timezone.utc).isoformat()},indent=2))
STATE_PY
}
trap record_state EXIT
export SETUP_PHASE=installing
installed=false
for attempt in 1 2 3 4 5 6; do
  printf 'Environment installation attempt %s at %s\n' "$attempt" "$(date -u +%FT%TZ)"
  if bash "$scripts/bootstrap_env.sh"; then installed=true; break; fi
  printf 'Attempt failed; verified download blocks remain available.\n'
  sleep 15
done
if [[ "$installed" != true ]]; then
  export SETUP_PHASE=blocked_installation
  exit 1
fi
source "$RESEARCH_ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$GS_ENV"
export PIP_REQUIRE_VIRTUALENV=false
export SETUP_PHASE=validating
python -m pip check
python "$scripts/preflight.py" --require-ready > "$SETUP_JOB_DIR/preflight.json"
python "$FOURDGS/train.py" --help > "$SETUP_JOB_DIR/train-help.txt"
# GPU 0 only. Do not reserve or stop any process; recheck immediately before launch.
for check in 1 2; do
  if ! python - <<'GPU_PY'
import subprocess
s=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
memory,utilization=map(int,s.strip().split(','))
if memory>=1024 or utilization>=10: raise SystemExit(1)
GPU_PY
  then
    export SETUP_PHASE=environment_ready_waiting_for_gpu
    exit 0
  fi
  if [[ "$check" == 1 ]]; then sleep 20; fi
done
CUDA_VISIBLE_DEVICES=0 python "$scripts/verify_env.py" > "$SETUP_JOB_DIR/cuda-validation.json"
export SETUP_PHASE=training_baseline
run_id="baseline-$(basename "$SETUP_JOB_DIR")"
python "$scripts/baseline.py" --run --gpu 0 --run-id "$run_id"
export SETUP_PHASE=baseline_completed
printf 'Baseline completed: %s\n' "$RESEARCH_ROOT/outputs/dnerf/bouncingballs/$run_id"
