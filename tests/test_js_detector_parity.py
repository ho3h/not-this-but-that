"""The demo's JS detectors must flag exactly what the Python v3 detector flags.

Runs the detector block from web/demo/app.js and web/demo/playground.js under
node over every committed generation (Q5b, Q5c, Q5d, both confirmations, the
baked demo examples) and compares family and cousin presence with
classifier.detect_v3. Skipped when node isn't installed.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

from classifier.detect_v3 import detect_v3

REPO = pathlib.Path(__file__).resolve().parent.parent
GENERATIONS = [
    "reports/q5c_d2_high_power.json",
    "reports/q5b_d1_continuation.json",
    "reports/q5d_minimal_set_d1_n120.json",
    "reports/confirmation_run.json",
]
JSONL = ["reports/confirmation_v2_generations.jsonl"]
# one answer per row, in a "text" field
TEXT_JSONL = ["reports/confirmation_v3_generations.jsonl", "reports/confirmation_v4_generations.jsonl",
              "reports/turn_up_generations.jsonl", "reports/confirmation_v5_gemma2_generations.jsonl",
              "reports/confirmation_v5_gemma3_generations.jsonl", "reports/g3_baseline_generations.jsonl"]

RUNNER = r"""
const fs = require('fs');
const [jsFile, textsFile] = process.argv.slice(1);
const src = fs.readFileSync(jsFile, 'utf8');
const block = src.slice(src.indexOf('  const CURLY_MAP'), src.indexOf('  function verdictText'));
const tiers = new Function(block + '\n return constructionTiers;')();
const texts = JSON.parse(fs.readFileSync(textsFile, 'utf8'));
process.stdout.write(JSON.stringify(texts.map(t => { const x = tiers(t); return [x.family, x.cousin]; })));
"""


def _texts() -> list[str]:
    texts = []
    for rel in GENERATIONS:
        for row in json.loads((REPO / rel).read_text())["rows"]:
            texts += [row["baseline"], row["ablated"]]
    for rel in JSONL:
        for line in (REPO / rel).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                texts += [row["baseline"], row["coalition"]]
    for rel in TEXT_JSONL:
        if (REPO / rel).exists():
            texts += [json.loads(l)["text"] for l in (REPO / rel).read_text().splitlines() if l.strip()]
    for ex in json.loads((REPO / "web/demo/examples.json").read_text())["examples"]:
        texts += [run["completion"] for run in ex["runs"].values()]
    return texts


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("js_file", ["web/demo/app.js", "web/demo/playground.js"])
def test_js_matches_python_family(tmp_path, js_file):
    texts = _texts()
    texts_file = tmp_path / "texts.json"
    texts_file.write_text(json.dumps(texts))
    out = subprocess.run(
        ["node", "-e", RUNNER, str(REPO / js_file), str(texts_file)],
        capture_output=True, text=True, check=True,
    )
    js = json.loads(out.stdout)
    py = []
    for t in texts:
        r = detect_v3(t)
        py.append([bool(r["family"]), bool(r["cousin"])])
    mismatches = [(t[:120], a, b) for t, a, b in zip(texts, py, js) if a != b]
    assert not mismatches, f"{len(mismatches)} JS/Python disagreements, e.g. {mismatches[:3]}"
