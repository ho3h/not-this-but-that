#!/bin/bash
# Pipeline orchestrator. Runs the full Q5c → per-layer attribution → ladder
# sequence as a single detached process. Survives bash kills via nohup.
#
# Usage:
#   nohup scripts/pipeline.sh > /tmp/pipeline.log 2>&1 &
#   disown
#
# Or, ergonomically:
#   scripts/pipeline.sh detach    # nohups itself, returns immediately
#   scripts/pipeline.sh status    # status snapshot
#   scripts/pipeline.sh           # foreground run (rare; debugging)

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PIPELINE_PID=/tmp/pipeline.pid
STATUS=/tmp/pipeline_status.txt
LOG=/tmp/pipeline.log

if [ "$1" = "detach" ]; then
    nohup "$0" run > "$LOG" 2>&1 &
    echo $! > "$PIPELINE_PID"
    disown $! 2>/dev/null || true
    echo "pipeline detached PID $(cat $PIPELINE_PID); log=$LOG status=$STATUS"
    exit 0
fi

if [ "$1" = "status" ]; then
    if [ -f "$PIPELINE_PID" ] && kill -0 "$(cat $PIPELINE_PID)" 2>/dev/null; then
        echo "RUNNING PID=$(cat $PIPELINE_PID)"
    else
        echo "NOT RUNNING"
    fi
    echo "--- status file ---"
    cat "$STATUS" 2>/dev/null || echo "(no status yet)"
    echo "--- last 10 log lines ---"
    tail -10 "$LOG" 2>/dev/null || echo "(no log yet)"
    exit 0
fi

# Default behaviour: run the pipeline now (foreground or via 'run' arg)
mark() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"
}
: > "$STATUS"  # truncate

mark "PIPELINE START"

# === Step 1: wait for Q5c runner ===
mark "STEP 1: wait for Q5c runner"
if [ -f /tmp/q5c.pid ]; then
    Q5C_PID=$(cat /tmp/q5c.pid)
    while kill -0 "$Q5C_PID" 2>/dev/null; do
        sleep 30
    done
    mark "  Q5c runner exited"
else
    mark "  no Q5c runner pidfile, assuming already done"
fi
# Summarise Q5c result
mark "  Q5c final state:"
.venv/bin/python -c "
import json, pathlib
d = json.loads(pathlib.Path('reports/q5c_d2_high_power.json').read_text())
rows = d['rows']
n = len(rows)
b = sum(1 for r in rows if r['baseline_core'])
a = sum(1 for r in rows if r['ablated_core'])
print(f'    pairs={n}  base={b}/{n}={b/n:.2%}  abl={a}/{n}={a/n:.2%}')
" | tee -a "$STATUS"

# === Step 2: stop daemon so attribution scans get full MPS ===
mark "STEP 2: stop daemon"
scripts/probe_run.sh stop 2>&1 | tee -a "$STATUS"
sleep 3

# === Step 3: attribution scan at L12 ===
mark "STEP 3: pivot_attribution at L12"
HF_TOKEN=$(grep -E '^HF_TOKEN=' .env 2>/dev/null | cut -d= -f2-) \
    .venv/bin/python scripts/pivot_attribution.py \
    --sae-layer 12 --top-k 100 --max-prompts 40 --checkpoint-every 10 \
    2>&1 | tail -15 | tee -a "$STATUS"

# === Step 4: attribution scan at L25 ===
mark "STEP 4: pivot_attribution at L25"
HF_TOKEN=$(grep -E '^HF_TOKEN=' .env 2>/dev/null | cut -d= -f2-) \
    .venv/bin/python scripts/pivot_attribution.py \
    --sae-layer 25 --top-k 100 --max-prompts 40 --checkpoint-every 10 \
    2>&1 | tail -15 | tee -a "$STATUS"

# === Step 5: restart daemon ===
mark "STEP 5: restart daemon"
scripts/probe_run.sh start 2>&1 | tee -a "$STATUS"
sleep 3

# === Step 6: ladder at L12 ===
mark "STEP 6: ladder at L12"
.venv/bin/python scripts/q7full_ladder.py --layer 12 --max-samples 80 --n-random 3 \
    2>&1 | tee -a "$STATUS"

# === Step 7: ladder at L25 ===
mark "STEP 7: ladder at L25"
.venv/bin/python scripts/q7full_ladder.py --layer 25 --max-samples 80 --n-random 3 \
    2>&1 | tee -a "$STATUS"

# === Step 8: final summary ===
mark "PIPELINE COMPLETE"
mark "Artifacts:"
mark "  reports/q5c_d2_high_power.json"
mark "  reports/pivot_attribution_L12.json"
mark "  reports/pivot_attribution_L25.json"
mark "  reports/asymptote_ladder_L12.json"
mark "  reports/asymptote_ladder_L25.json"
