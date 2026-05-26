#!/bin/bash
# Detached sequencer: waits for q5d, then runs q5b, then q5c, then M2 ladder.
# Each step writes its own log. Status reported to /tmp/seq_status.txt.

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

STATUS=/tmp/seq_status.txt
: > "$STATUS"

mark() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

mark "SEQUENCER START"

# === Step 1: wait for q5d ===
mark "STEP 1: wait for q5d"
if [ -f /tmp/q5d.pid ]; then
    while kill -0 "$(cat /tmp/q5d.pid)" 2>/dev/null; do
        sleep 30
    done
    mark "  q5d exited"
fi
tail -3 /tmp/q5d_n120.log 2>/dev/null | tee -a "$STATUS"

# === Step 2: q5b expansion ===
mark "STEP 2: launch q5b expansion (n=300 pairs)"
.venv/bin/python -u scripts/q5b_runner.py 2>&1 | tail -30 | tee -a "$STATUS"
mark "  q5b done"

# === Step 3: q5c full D2 ===
mark "STEP 3: launch q5c full-D2 expansion (n=306 pairs)"
.venv/bin/python -u scripts/q5c_runner.py 2>&1 | tail -30 | tee -a "$STATUS"
mark "  q5c done"

# === Step 4: M2 ladder at n=200 ===
mark "STEP 4: M2 ladder at n=200 (all D1)"
.venv/bin/python -u <<'PY' 2>&1 | tail -30 | tee -a "$STATUS"
import json, urllib.request, pathlib, time
PROBE = "http://127.0.0.1:8765/probe"
def call(body, timeout=3600):
    req = urllib.request.Request(PROBE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# Pull attribution-ranked sets at every size, from CURRENT pivot_attribution.json
# (which may be the n=100 version by now).
sizes = [5, 10, 25, 50, 75, 100]
sets = {}
for n in sizes:
    r = call({"cmd": "attribution", "top_n": n, "kind": "promote"})["result"]
    sets[f"attrib_top{n}"] = r["features"]

print(f"running ladder at max_samples=200 (all D1) with 3 random draws/size…")
t0 = time.perf_counter()
resp = call({
    "cmd": "ladder",
    "conditions": sets,
    "n_random_per_size": 3,
    "variants": ["C1", "C2", "C3"],
    "max_samples": 200,
    "seed": 11,
})
print(f"done in {time.perf_counter()-t0:.0f}s")
out = pathlib.Path("reports/asymptote_ladder_n200.json")
out.write_text(json.dumps(resp, indent=2))
r = resp["result"]
print(f"baseline P(pivot) @ n={r['n_samples']}: {r['baseline_mean_p_pivot']:.4f}")
for n in sizes:
    s = r["by_condition"][f"attrib_top{n}"]
    print(f"  attrib_top{n:>3}: n={s['n_features']:>3} drop={s['mean_drop']:+.4f} rel={s['rel_drop']:+.2%}")
print(f"→ {out}")
PY

mark "SEQUENCER COMPLETE"
mark "Artifacts:"
mark "  reports/q5d_minimal_set_d1_n120.json"
mark "  reports/q5b_d1_continuation.json (expanded)"
mark "  reports/q5c_d2_high_power.json (expanded)"
mark "  reports/asymptote_ladder_n200.json"
mark "  reports/pivot_attribution.json (n=100, if attribution finished)"
