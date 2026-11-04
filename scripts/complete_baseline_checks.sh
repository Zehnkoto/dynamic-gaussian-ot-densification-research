#!/usr/bin/env bash
# Persistent continuation: only writes under the owner data volume.
set -euo pipefail
scripts=$(dirname "$(realpath "$0")")
source "$scripts/common.sh"
: "${CHECK_JOB_DIR:?Missing job directory}"
export CHECK_PHASE=starting
record_state() {
  export CHECK_EXIT="$?"
  "$GS_ENV/bin/python" -B - <<'STATE'
import os,json
from pathlib import Path
from datetime import datetime,timezone
(Path(os.environ['CHECK_JOB_DIR'])/'state.json').write_text(json.dumps({'phase':os.environ['CHECK_PHASE'],'exit_code':int(os.environ['CHECK_EXIT']),'updated_utc':datetime.now(timezone.utc).isoformat()},indent=2))
STATE
}
trap record_state EXIT
export CHECK_PHASE=downloading_weights
record_state
ready=false
for attempt in 1 2 3 4 5 6 7 8; do
  echo "Weight resume attempt $attempt"
  if "$GS_ENV/bin/python" -B "$scripts/fetch_eval_weights.py" --net alex; then ready=true; break; fi
  echo 'Download interrupted; preserving all received chunks.'
  sleep 10
done
if [[ "$ready" != true ]]; then export CHECK_PHASE=download_failed; exit 1; fi
export CHECK_PHASE=evaluating_lpips
record_state
# Use an idle GPU with ample free memory, or CPU when none is available.
device=cpu
selected=$("$GS_ENV/bin/python" -B - <<'GPU'
import subprocess
try:
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 candidates=[tuple(map(int,line.split(','))) for line in s.splitlines()]
 candidates=[row for row in candidates if row[1]>=8192 and row[2]<10]
 print(max(candidates,key=lambda row:row[1])[0] if candidates else '')
except (OSError,subprocess.SubprocessError):print('')
GPU
)
if [[ -n "$selected" ]]; then export CUDA_VISIBLE_DEVICES="$selected"; device=cuda; fi
"$GS_ENV/bin/python" -B "$scripts/evaluate_lpips.py"   --model "$RESEARCH_ROOT/outputs/dnerf/bouncingballs/baseline-numpy1235-20261030T174401Z"   --output "$CHECK_JOB_DIR/lpips-alex.json" --device "$device"
export CHECK_PHASE=completed
record_state
echo 'Baseline LPIPS evaluation completed.'
