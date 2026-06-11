// not-this-but-that — demo v2
// Narrative scroll. Pre-baked side-by-side playback. Live daemon mode.
// Defensive loading: examples.json first (always works), layout.json + daemon
// optional. If the daemon is down, baked playback still works end-to-end.

(function () {
  "use strict";

  const PROBE = "/probe";
  const LAYOUT_URL = "/demo/layout.json";
  const EXAMPLES_URL = "/demo/examples.json";

  let LAYOUT = null;       // { features: [{idx, x, y, cid, density, label}] }
  let EXAMPLES = null;     // { top25_set, minimal_set, examples: [...] }
  let CANVAS, CTX, DPR = window.devicePixelRatio || 1;
  let ACTIVITY = null;     // Float32Array(16384), per-frame decaying
  let ABLATED = new Set(); // currently-flagged-for-ablation feature ids

  let currentExampleIdx = 0;
  let currentSet = "top25";   // "top25" | "minimal"
  let playSession = 0;        // bumped on every Play to cancel stale runs

  // ─── helpers ────────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function setStatus(state, msg) {
    const el = $("status");
    el.className = state === "up" ? "pill-up" : "pill-down";
    el.textContent = msg;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  async function init() {
    // 1. Load examples (required for the headline experience)
    try {
      EXAMPLES = await fetch(EXAMPLES_URL).then(r => r.json());
      renderExampleGrid();
    } catch (e) {
      console.warn("examples.json failed:", e);
      $("example-grid").innerHTML = '<div style="color: var(--warn); padding: 20px;">Pre-baked examples not available. Try the live mode at the bottom.</div>';
    }

    // 2. Load layout (for the cloud — can fail without breaking the page)
    try {
      LAYOUT = await fetch(LAYOUT_URL).then(r => r.json());
      ACTIVITY = new Float32Array(LAYOUT.features.length);
      setupCanvas();
      redraw();
    } catch (e) {
      console.warn("layout.json failed:", e);
      const stage = document.querySelector(".map-stage");
      if (stage) stage.style.opacity = 0.4;
    }

    // 3. Wire up controls (works regardless of daemon)
    wireControls();

    // 4. Optional: ping the daemon for the status pill
    let daemonOk = false;
    try {
      const resp = await fetch(PROBE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "ping" }),
      });
      const d = await resp.json();
      if (d.ok) { setStatus("up", `daemon UP (Gemma 2 2B on ${d.result.device})`); daemonOk = true; }
      else setStatus("down", "daemon: " + (d.error || "unknown"));
    } catch (e) {
      setStatus("down", "daemon down — playground live mode disabled; baked examples still work");
    }
    // Article-head status pill (graphgeometry-style "data ready")
    const apill = $("status-pill");
    if (apill) {
      if (EXAMPLES && daemonOk) {
        apill.setAttribute("data-status", "up");
        apill.textContent = `Live data · ${EXAMPLES.examples.length} baked examples + daemon connected`;
      } else if (EXAMPLES) {
        apill.setAttribute("data-status", "up");
        apill.textContent = "Baked examples loaded · daemon offline";
      } else {
        apill.setAttribute("data-status", "down");
        apill.textContent = "Couldn't load examples";
      }
    }
  }

  // ─── example grid ───────────────────────────────────────────────────────
  function renderExampleGrid() {
    const grid = $("example-grid");
    grid.innerHTML = "";
    EXAMPLES.examples.forEach((ex, i) => {
      const card = document.createElement("div");
      card.className = "example-card" + (i === currentExampleIdx ? " active" : "");
      card.dataset.idx = i;
      card.innerHTML = `
        <div class="ck">Example ${i + 1}</div>
        <div class="pp">${esc(ex.label)}</div>
        <div class="meta">${ex.runs.baseline.records.length} tokens · seed ${ex.seed}</div>
      `;
      card.addEventListener("click", () => {
        currentExampleIdx = i;
        document.querySelectorAll(".example-card").forEach(c => c.classList.remove("active"));
        card.classList.add("active");
        renderStaticPreview();
      });
      grid.appendChild(card);
    });
    renderStaticPreview();
  }

  function renderStaticPreview() {
    // Just show the static completion text on both sides, with highlights.
    const ex = EXAMPLES.examples[currentExampleIdx];
    const right = currentSet === "top25" ? ex.runs.top25 : ex.runs.minimal;
    $("text-left").innerHTML = highlightConstruction(ex.runs.baseline.completion);
    $("text-right").innerHTML = highlightConstruction(right.completion);
    $("ablated-label").textContent = right.ablate_label;
    setVerdict("verdict-left", ex.runs.baseline.completion);
    setVerdict("verdict-right", right.completion);
    $("play-meta").textContent = `prompt: "${ex.prompt}"`;
  }

  function setVerdict(elId, text) {
    const has = constructionHits(text);
    const el = $(elId);
    if (has.length) {
      el.className = "verdict hit";
      el.innerHTML = `✓ construction present (${has.join(", ")})`;
    } else {
      el.className = "verdict clean";
      el.innerHTML = "· no construction detected";
    }
  }

  // ─── visual construction detector ───────────────────────────────────────
  // More permissive than the canonical Python classifier — catches F2
  // staccato (cross-sentence) and F1/F3 (same-sentence) so the UI shows the
  // construction users can actually SEE in the prose.
  // 2026-06-09: mirrors the fixed Python permissive layer in
  // src/classifier/detect_v2.py — negation mandatory, \b on pronoun tails,
  // first-person/epistemic/bare-do-support FP classes excluded. See
  // reports/permissive_fix_audit.md.
  const PATTERNS = [
    {
      name: "F1/F3 (same-sentence)",
      re: /\b(?:(?:is|are|was|were)n'?t|(?:is|are|was|were|am)\s+not|(?:it|that|this|he|she|there)'?s\s+not|(?:we|they|you)'?re\s+not|(?:do|does|did)(?:n'?t|\s+not)\s+(?:just|merely|simply|only|necessarily|really|mean))(?!\s+(?:sure|certain|aware|convinced)\b)\s+(?:just\s+|merely\s+|simply\s+|only\s+)?[^.,;:!?\n]{1,80}[,;—–\-]\s*(?:it'?s?|that'?s|they'?re|they|he'?s?|she'?s?|we'?re|we|but|rather|instead)\b/gi,
    },
    {
      name: "F2 staccato (cross-sentence)",
      re: /\b(?:(?:is|are|was|were)n'?t|(?:is|are|was|were|am)\s+not|(?:[Ii]t|[Tt]hat|[Tt]his|[Hh]e|[Ss]he|[Tt]here)'?s\s+not|(?:[Ww]e|[Tt]hey|[Yy]ou)'?re\s+not|(?:[Dd]o|[Dd]oes|[Dd]id)(?:n'?t|\s+not)\s+(?:just|merely|simply|only|necessarily|really|mean))(?!\s+(?:sure|certain|aware|convinced)\b)\s+(?:just\s+|merely\s+|simply\s+|only\s+|about\s+)?[^.!?\n]{1,80}[.!?]\s*(?:It'?s?|That'?s|They'?re|They|He'?s?|She'?s?|We'?re|We|But|Rather|Instead)\b/g,
    },
    {
      name: "F4/F5 (less/more, not about)",
      re: /(?:\bless\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*more\b|\bnot\s+about\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*(?:it'?s?\s+about|about))/gi,
    },
  ];

  function constructionHits(text) {
    const hits = [];
    for (const p of PATTERNS) {
      p.re.lastIndex = 0;
      if (p.re.test(text)) hits.push(p.name);
    }
    return hits;
  }

  function highlightConstruction(text) {
    let out = esc(text);
    // Apply highlights for each pattern; replace plain text with span-wrapped.
    for (const p of PATTERNS) {
      p.re.lastIndex = 0;
      out = out.replace(p.re, m => `<span class="construction">${m}</span>`);
    }
    return out;
  }

  // ─── intervention picker ────────────────────────────────────────────────
  function setIntervention(set) {
    currentSet = set;
    document.querySelectorAll(".btn-chip[data-set]").forEach(b => {
      b.classList.toggle("on", b.dataset.set === set);
    });
    renderStaticPreview();
  }

  // ─── playback (side-by-side, lockstep) ──────────────────────────────────
  async function play() {
    if (!EXAMPLES) return;
    const ex = EXAMPLES.examples[currentExampleIdx];
    const rightRun = currentSet === "top25" ? ex.runs.top25 : ex.runs.minimal;
    const ms = parseInt($("speed").value, 10);

    playSession += 1;
    const mySession = playSession;
    $("play").disabled = true;
    $("play").textContent = "▶ Playing…";

    // Reset
    $("text-left").innerHTML = "";
    $("text-right").innerHTML = "";
    $("verdict-left").innerHTML = "";
    $("verdict-right").innerHTML = "";
    if (ACTIVITY) ACTIVITY.fill(0);
    ABLATED = new Set(rightRun.ablate_set);  // visually show ablated on cloud
    startAnim();

    const leftRecs = ex.runs.baseline.records;
    const rightRecs = rightRun.records;
    const maxLen = Math.max(leftRecs.length, rightRecs.length);
    const leftBuf = [], rightBuf = [];

    for (let i = 0; i < maxLen; i++) {
      if (mySession !== playSession) break;  // cancelled
      $("hud_step").textContent = `step ${i + 1} / ${maxLen}`;
      if (i < leftRecs.length) {
        leftBuf.push(leftRecs[i].token_str);
        $("text-left").innerHTML = highlightConstruction(leftBuf.join(""));
        // Spike the cloud with the RIGHT-side features (the run that's
        // actually different from baseline) so the ablation effect is visible
        for (const f of (rightRecs[i] && rightRecs[i].top_features) || []) {
          ACTIVITY[f.idx] = 1.0;
        }
      }
      if (i < rightRecs.length) {
        rightBuf.push(rightRecs[i].token_str);
        $("text-right").innerHTML = highlightConstruction(rightBuf.join(""));
      }
      await sleep(ms);
    }

    if (mySession === playSession) {
      setVerdict("verdict-left", leftBuf.join(""));
      setVerdict("verdict-right", rightBuf.join(""));
      $("play").disabled = false;
      $("play").textContent = "▶ Play again";
      // Let cloud fade for a beat
      setTimeout(stopAnim, 2000);
    }
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ─── cloud rendering ────────────────────────────────────────────────────
  function setupCanvas() {
    CANVAS = $("canvas");
    CTX = CANVAS.getContext("2d");
    function resize() {
      const r = CANVAS.getBoundingClientRect();
      CANVAS.width = r.width * DPR;
      CANVAS.height = r.height * DPR;
      redraw();
    }
    window.addEventListener("resize", resize);
    resize();
    CANVAS.addEventListener("mousemove", onHover);
    CANVAS.addEventListener("mouseleave", () => { $("tip").style.display = "none"; });
    CANVAS.addEventListener("click", onClick);
  }

  function xy(f) {
    const W = CANVAS.width, H = CANVAS.height;
    const m = Math.min(W, H) * 0.46;
    return [W / 2 + f.x * m, H / 2 + f.y * m];
  }

  // 18 communities, give each a hue. The "instruction-following" community
  // (cid=12) where 3223 sits gets a deliberate gold to telegraph importance.
  const COMMUNITY_HUE = {};
  function commColor(cid, alpha = 0.5) {
    if (cid === 12) return `rgba(255, 212, 121, ${alpha})`; // 3223's community
    if (!(cid in COMMUNITY_HUE)) {
      // golden-angle hue distribution
      const k = Object.keys(COMMUNITY_HUE).length;
      COMMUNITY_HUE[cid] = (k * 137.508) % 360;
    }
    return `hsla(${COMMUNITY_HUE[cid]}, 35%, 55%, ${alpha})`;
  }

  function redraw() {
    if (!LAYOUT || !CANVAS) return;
    const W = CANVAS.width, H = CANVAS.height;
    CTX.clearRect(0, 0, W, H);
    if (ACTIVITY) for (let i = 0; i < ACTIVITY.length; i++) ACTIVITY[i] *= 0.94;

    const feats = LAYOUT.features;
    for (let i = 0; i < feats.length; i++) {
      const f = feats[i];
      const [x, y] = xy(f);
      const act = ACTIVITY ? ACTIVITY[i] : 0;
      let color, r;
      if (ABLATED.has(i)) {
        color = "#ff8d6b";
        r = 2.5 * DPR;
      } else if (act > 0.7) {
        color = "#ffd479";
        r = 3.2 * DPR;
      } else if (act > 0.05) {
        const a = 0.35 + 0.65 * act;
        color = `rgba(155, 227, 168, ${a.toFixed(2)})`;
        r = (1.6 + 1.3 * act) * DPR;
      } else {
        color = commColor(f.cid, 0.18);
        r = 0.85 * DPR;
      }
      CTX.fillStyle = color;
      CTX.beginPath();
      CTX.arc(x, y, r, 0, Math.PI * 2);
      CTX.fill();
    }
    // Halos on hot
    if (ACTIVITY) {
      for (let i = 0; i < ACTIVITY.length; i++) {
        if (ACTIVITY[i] > 0.7 && !ABLATED.has(i)) {
          const [x, y] = xy(feats[i]);
          CTX.strokeStyle = "rgba(255, 212, 121, 0.45)";
          CTX.lineWidth = 1.2 * DPR;
          CTX.beginPath();
          CTX.arc(x, y, 6 * DPR, 0, Math.PI * 2);
          CTX.stroke();
        }
      }
    }
  }

  let animHandle = null;
  function startAnim() {
    if (animHandle) return;
    (function loop() { redraw(); animHandle = requestAnimationFrame(loop); })();
  }
  function stopAnim() {
    if (animHandle) { cancelAnimationFrame(animHandle); animHandle = null; }
  }

  function onHover(ev) {
    if (!LAYOUT) return;
    const rect = CANVAS.getBoundingClientRect();
    const mx = (ev.clientX - rect.left) * DPR;
    const my = (ev.clientY - rect.top) * DPR;
    let nearest = null, nd = 1e9;
    const feats = LAYOUT.features;
    for (let i = 0; i < feats.length; i++) {
      const [x, y] = xy(feats[i]);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < nd) { nd = d; nearest = i; }
    }
    const r = 10 * DPR;
    if (nearest !== null && nd < r * r) {
      const f = feats[nearest];
      const tip = $("tip");
      tip.style.display = "block";
      tip.style.left = (ev.clientX - CANVAS.getBoundingClientRect().left + 14) + "px";
      tip.style.top = (ev.clientY - CANVAS.getBoundingClientRect().top + 10) + "px";
      tip.innerHTML = `<span class="idx">feat ${f.idx}</span> · community ${f.cid}<br>${esc(f.label) || "(no label)"}<div class="meta">density ${(f.density*100).toFixed(1)}% · click to ${ABLATED.has(f.idx) ? "un-" : ""}ablate</div>`;
    } else {
      $("tip").style.display = "none";
    }
  }

  function onClick(ev) {
    if (!LAYOUT) return;
    const rect = CANVAS.getBoundingClientRect();
    const mx = (ev.clientX - rect.left) * DPR;
    const my = (ev.clientY - rect.top) * DPR;
    let nearest = null, nd = 1e9;
    const feats = LAYOUT.features;
    for (let i = 0; i < feats.length; i++) {
      const [x, y] = xy(feats[i]);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < nd) { nd = d; nearest = i; }
    }
    const r = 12 * DPR;
    if (nearest !== null && nd < r * r) {
      if (ABLATED.has(nearest)) ABLATED.delete(nearest);
      else ABLATED.add(nearest);
      redraw();
    }
  }

  // ─── live (daemon) mode ─────────────────────────────────────────────────
  async function liveRun() {
    const prompt = $("custom_prompt").value.trim();
    if (!prompt) return;
    const max_new = parseInt($("custom_max").value, 10);
    const seed = parseInt($("custom_seed").value, 10);
    const ablateSet = currentSet === "top25" ? EXAMPLES.top25_set : EXAMPLES.minimal_set;

    $("live-run").disabled = true;
    $("live-meta").textContent = "calling daemon (5-15s)…";

    try {
      // Two calls in parallel: baseline + ablated. Daemon serializes so it
      // ends up sequential, but the UX is one click.
      const [baseR, ablR] = await Promise.all([
        fetch(PROBE, { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cmd: "generate_with_activations", model: "it",
            prompt, ablate: null, max_new_tokens: max_new, seed, top_k_features: 10 }) }).then(r => r.json()),
        fetch(PROBE, { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cmd: "generate_with_activations", model: "it",
            prompt, ablate: ablateSet, max_new_tokens: max_new, seed, top_k_features: 10 }) }).then(r => r.json()),
      ]);

      if (!baseR.ok || !ablR.ok) {
        $("live-meta").textContent = "error: " + (baseR.error || ablR.error || "unknown");
        return;
      }
      // Inject as a synthetic example and play it
      EXAMPLES.examples.unshift({
        id: "live", label: `LIVE: "${prompt.slice(0, 40)}…"`, prompt,
        max_new_tokens: max_new, seed,
        runs: {
          baseline: { ablate_label: "no intervention", ablate_set: [],
                       completion: baseR.result.completion, records: baseR.result.records },
          top25:    { ablate_label: "top-25 coalition (live)", ablate_set: ablateSet,
                       completion: ablR.result.completion, records: ablR.result.records },
          minimal:  { ablate_label: "minimal core (live, reused top-25)", ablate_set: ablateSet,
                       completion: ablR.result.completion, records: ablR.result.records },
        },
      });
      currentExampleIdx = 0;
      renderExampleGrid();
      $("live-meta").textContent = `done — playing back (baseline took ${baseR.elapsed_s.toFixed(1)}s, ablated ${ablR.elapsed_s.toFixed(1)}s)`;
      play();
    } catch (e) {
      $("live-meta").textContent = "error: " + e.message;
    } finally {
      $("live-run").disabled = false;
    }
  }

  // ─── wiring ─────────────────────────────────────────────────────────────
  function wireControls() {
    $("play").addEventListener("click", play);
    document.querySelectorAll(".btn-chip[data-set]").forEach(b => {
      b.addEventListener("click", () => setIntervention(b.dataset.set));
    });
  }

  init().catch(e => {
    console.error("init error:", e);
    setStatus("down", "init error: " + e.message);
  });
})();
