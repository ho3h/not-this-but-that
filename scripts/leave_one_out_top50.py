"""Extend Q3 leave-one-out from top-25 to top-50.

Same methodology: for each of 50 features, ablate the *other* 49, measure
P(pivot), record the cost (= full-set drop − this-set drop). High cost =
indispensable. Low cost = substitutable.

Tells us whether the 2 indispensable + 22 substitutable structure persists
at 50, or whether new indispensables appear between ranks 26 and 50.

Writes reports/q3b_leave_one_out_top50.json. Resumable.
"""
from __future__ import annotations
import json, pathlib, time, urllib.request

PROBE = "http://127.0.0.1:8765/probe"
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "q3b_leave_one_out_top50.json"


def call(body, timeout=600):
    req = urllib.request.Request(PROBE, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    top50 = call({"cmd": "attribution", "top_n": 50, "kind": "promote"})["result"]["features"]
    labels = call({"cmd": "labels", "features": top50})["result"]
    print(f"top-50: {top50[:8]}…")

    if OUT.exists():
        state = json.loads(OUT.read_text())
        if state.get("top50") != top50:
            print(f"  top50 changed since last run, restarting")
            state = None
    else:
        state = None
    if state is None:
        print(f"\nMeasuring full top-50 drop…")
        full = call({"cmd": "measure_pivot", "ablate": top50, "max_samples": 80})["result"]
        state = {
            "top50": top50,
            "baseline_mean_p_pivot": full["baseline_mean"],
            "full_set_drop": full["mean_drop"],
            "full_set_rel_drop": full["rel_drop"],
            "leave_one_out": [None] * 50,
        }
        OUT.write_text(json.dumps(state, indent=2))
        print(f"  baseline={state['baseline_mean_p_pivot']:.4f} "
              f"full_drop={state['full_set_drop']:+.4f} ({state['full_set_rel_drop']:+.2%})")

    t0 = time.perf_counter()
    for i in range(50):
        if state["leave_one_out"][i] is not None:
            continue
        f = top50[i]
        test_set = [x for x in top50 if x != f]
        r = call({"cmd": "measure_pivot", "ablate": test_set, "max_samples": 80})["result"]
        cost = state["full_set_drop"] - r["mean_drop"]
        state["leave_one_out"][i] = {
            "removed": f, "label": labels.get(str(f), ""),
            "drop_without": r["mean_drop"], "rel_drop_without": r["rel_drop"],
            "cost_of_removal": cost,
        }
        OUT.write_text(json.dumps(state, indent=2))
        elapsed = time.perf_counter() - t0
        print(f"  [{i+1:>2}/50] remove {f:>5}  cost={cost:+.5f}  drop={r['mean_drop']:+.4f}  "
              f"{labels.get(str(f), '')[:50]}  (cum {elapsed:.0f}s)")

    # Summary
    print(f"\nDone in {time.perf_counter() - t0:.0f}s.")
    ranked = sorted(state["leave_one_out"], key=lambda r: -r["cost_of_removal"])
    print(f"\nTop-10 most indispensable (highest cost when removed):")
    for r in ranked[:10]:
        print(f"  feat {r['removed']:>5}  cost={r['cost_of_removal']:+.5f}  {r['label'][:55]}")


if __name__ == "__main__":
    main()
