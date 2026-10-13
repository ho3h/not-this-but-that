// not-this-but-that — demo v2
// Narrative scroll. Pre-baked side-by-side playback: normal vs the 25 off.
// Defensive loading: examples.json first (always works), layout.json optional.

(function () {
  "use strict";

  const LAYOUT_URL = "/demo/layout.json";
  const EXAMPLES_URL = "/demo/examples.json";

  let LAYOUT = null;       // { features: [{idx, x, y, cid, density, label}] }
  let EXAMPLES = null;     // { off_set, examples: [{ runs: { baseline, off } }] }
  let CANVAS, CTX, DPR = window.devicePixelRatio || 1;
  let ACTIVITY = null;     // Float32Array(16384), per-frame decaying
  let ABLATED = new Set(); // currently-flagged-for-ablation feature ids

  let currentExampleIdx = 0;
  let playSession = 0;        // bumped on every Play to cancel stale runs

  // ─── helpers ────────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function setStatus(state, msg) {
    const el = $("status");
    if (!el) { console.warn(msg); return; }
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
      $("example-grid").innerHTML = '<div style="color: var(--warn); padding: 20px;">The recorded examples didn\'t load. Try reloading the page.</div>';
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

    // 3. Wire up controls
    wireControls();

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
    const right = ex.runs.off;
    $("text-left").innerHTML = highlightConstruction(ex.runs.baseline.completion);
    $("text-right").innerHTML = highlightConstruction(right.completion);
    $("ablated-label").textContent = right.ablate_label;
    setVerdict("verdict-left", ex.runs.baseline.completion);
    setVerdict("verdict-right", right.completion);
    $("play-meta").textContent = `Question: "${ex.prompt}"`;
  }

  function setVerdict(elId, text) {
    const v = verdictText(text);
    const el = $(elId);
    el.className = "verdict " + (v.hit ? "hit" : "clean");
    el.textContent = v.label;
  }


  // ─── construction detector (mirrors src/classifier/detect_v3.py) ─────────
  // Two tiers: the negated family ("isn't just X, it's Y" and relatives) and its
  // cousin ("more than just X. It's Y"). Curly quotes are normalised first;
  // v1 hits on epistemic hedges ("not sure ..., but") are dropped. app.js and
  // playground.js carry identical copies; tests/test_js_detector_parity.py
  // checks both against the Python detector on every saved answer.
  const CURLY_MAP = { "’": "'", "‘": "'", "“": '"', "”": '"' };
  function normaliseQuotes(t) { return t.replace(/[‘’“”]/g, c => CURLY_MAP[c]); }
  const V1_PIVOT = String.raw`(?:\b(?:it|that|this|he|she|we|they|these|those|there)'?s\b|\b(?:we|they|you)'?re\b|\b(?:he|she|it|this|that|there)\s+(?:is|was)\b|\b(?:we|they|these|those|you)\s+(?:are|were)\b)`;
  const V1_NEG = String.raw`(?:\b(?:is|are|was|were|am)\s+not\b|\b(?:is|are|was|were)n'?t\b|\b(?:it|that|this|he|she|we|they|these|those|there)'?s\s+not\b|\b(?:we|they|you)'?re\s+not\b)`;
  const NEG_OPENER = String.raw`(?:(?:is|are|was|were)n'?t|(?:is|are|was|were|am)\s+not|(?:[Ii]t|[Tt]hat|[Tt]his|[Hh]e|[Ss]he|[Tt]here)'?s\s+not|(?:[Ww]e|[Tt]hey|[Yy]ou)'?re\s+not|(?:[Dd]o|[Dd]oes|[Dd]id)(?:n'?t|\s+not)\s+(?:just|merely|simply|only|necessarily|really|mean))(?!\s+(?:sure|certain|aware|convinced)\b)`;
  const TAIL_SAME = String.raw`(?:it'?s?|that'?s|they'?re|they|he'?s?|she'?s?|we'?re|we|but|rather|instead)\b`;
  const TAIL_STACCATO = String.raw`(?:It'?s?|That'?s|They'?re|They|He'?s?|She'?s?|We'?re|We|But|Rather|Instead)\b`;
  const TAIL_V3 = String.raw`(?:It'?s?|That'?s|They'?re|They|He'?s?|She'?s?|We'?re|We|But|Rather|Instead|This|These|Those|There'?s?)\b`;
  const PIVOT_ANY = String.raw`(?:it|they|he|she|we|you|this|that|these|those)(?:'s|'re|\s+\w+)\b`;
  const COP = String.raw`(?:\b(?:is|are|was|were|be|been|being)|'s|'re)`;
  const DEGREE = String.raw`(?:(?:so\s+much|far|much|a\s+lot|a\s+whole\s+lot|far\s+much|way)\s+)?`;
  const MTJ_HEAD = String.raw`\bmore\s+than\s+(?:just|merely|simply)\b`;
  const EPISTEMIC = /^not\s+(?:sure|certain|aware|convinced|clear)\b/i;
  const R = (src, flags) => new RegExp(src, "g" + (flags || ""));
  const PATTERNS = [
    // family: v1 regex hinges (detect.py)
    { tier: "family", v1: true, re: R(String.raw`\bnot\s+only\b[^.!?\n]{1,120}?\b(?:but|but\s+also|yet|also|—|–|--)\b`, "i") },
    { tier: "family", v1: true, re: R(String.raw`\b(?:not|(?:is|are|was|were)n'?t)\s+(?:just|merely|simply)\b[^.!?\n]{1,120}?(?:[—–]|--|\bbut\b|` + V1_PIVOT + ")", "i") },
    { tier: "family", v1: true, re: R(V1_NEG + String.raw`[^.!?\n]{1,120}?[,;—–]\s*` + V1_PIVOT, "i") },
    { tier: "family", v1: true, re: R(String.raw`\bnot\s+(?:a|an|the|just|only|merely)?\s*[^.!?\n,]{1,80},?\s*but\b`, "i") },
    { tier: "family", v1: true, re: R(String.raw`\bno\s+\w[^.!?\n]{0,60}\.\s+no\s+\w[^.!?\n]{0,60}\.\s+(?:just|only|but)\b`, "i") },
    // family: permissive layer (detect_v2.py)
    { tier: "family", re: R(String.raw`\b` + NEG_OPENER + String.raw`\s+(?:just\s+|merely\s+|simply\s+|only\s+)?[^.,;:!?\n]{1,80}[,;—–\-]\s*` + TAIL_SAME, "i") },
    { tier: "family", re: R(String.raw`\b` + NEG_OPENER + String.raw`\s+(?:just\s+|merely\s+|simply\s+|only\s+|about\s+)?[^.!?\n]{1,80}[.!?]\s*` + TAIL_STACCATO) },
    { tier: "family", re: R(String.raw`(?:\bless\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*more\b|\bnot\s+about\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*(?:it'?s?\s+about|about))`, "i") },
    // family: v3 additions
    { tier: "family", re: R(String.raw`\b(This|That)\s+(?:isn't|wasn't|is\s+not|was\s+not)\s+(?:just\s+|merely\s+|simply\s+|about\s+)?[^.!?\n]{1,80}[.!?]\s*\1\s+(?:is|was|'s)\b`) },
    { tier: "family", re: R(String.raw`\bnot\s+(?:just|merely|only|simply)\b[^.!?\n;,—–]{1,80}[,;—–]\s*` + PIVOT_ANY, "i") },
    { tier: "family", re: R(String.raw`\bno\s+longer\s+(?:just|merely|simply)\b[^.!?\n]{1,80}[,;—–.]\s*(?:it'?s|they'?re|that'?s|it\s+is|they\s+are)\b`, "i") },
    { tier: "family", re: R(String.raw`\bless\s+(?:about|like|of)\b[^.!?\n]{1,80}?\b(?:and|but|,|;|—|–)?\s*more\s+(?:about|like|of)\b`, "i") },
    // cousin: v2
    { tier: "cousin", re: R(String.raw`\b(?:is|are|was|were)\s+(?:much\s+)?more\s+than\s+(?:just\s+|merely\s+|simply\s+)?[^.,;:!?\n]{1,80}[,;—–\-]\s*` + TAIL_SAME, "i") },
    { tier: "cousin", re: R(String.raw`\b(?:is|are|was|were)\s+(?:much\s+)?more\s+than\s+(?:just\s+|merely\s+|simply\s+)?[^.!?\n]{1,80}[.!?]\s*` + TAIL_STACCATO) },
    // cousin: v3 additions
    { tier: "cousin", re: R(COP + String.raw`\s+` + DEGREE + String.raw`more\s+than\s+(?:just\s+|merely\s+|simply\s+)?[^.,;:!?\n]{1,80}[,;—–\-]\s*(?:it'?s?|that'?s|they'?re|they|he'?s?|she'?s?|we'?re|we|but|rather|instead|this|these)\b`, "i") },
    { tier: "cousin", re: R(COP + String.raw`\s+` + DEGREE + String.raw`more\s+than\s+(?:just\s+|merely\s+|simply\s+)?[^.!?\n]{1,80}[.!?]\s*` + TAIL_V3) },
    { tier: "cousin", re: R(MTJ_HEAD + String.raw`[^.!?\n;,—–]{1,80}[,;—–]\s*` + PIVOT_ANY, "i") },
    { tier: "cousin", re: R(MTJ_HEAD + String.raw`[^.!?\n]{1,80}[.!?]\s*` + TAIL_V3) },
    { tier: "cousin", re: R(String.raw`,\s*more\s+than\s+(?:just|merely|simply)\b[^.!?\n,]{1,60},\s*(?:is|are|was|were|has|have|becomes?)\b`, "i") },
  ];

  function constructionSpans(text) {
    const t = normaliseQuotes(text);  // 1:1 character swaps, so spans index the original
    const spans = [];
    for (const p of PATTERNS) {
      p.re.lastIndex = 0;
      let m;
      while ((m = p.re.exec(t)) !== null) {
        if (!(p.v1 && EPISTEMIC.test(m[0]))) spans.push([m.index, m.index + m[0].length, p.tier]);
        if (m[0].length === 0) p.re.lastIndex++;
      }
    }
    return spans.sort((a, b) => a[0] - b[0]);
  }

  function constructionTiers(text) {
    const tiers = new Set(constructionSpans(text).map(s => s[2]));
    return { family: tiers.has("family"), cousin: tiers.has("cousin") };
  }

  function verdictText(text) {
    const t = constructionTiers(text);
    if (t.family) return { hit: true, label: "Uses “not X but Y”" };
    if (t.cousin) return { hit: true, label: "Uses the “more than just” variant" };
    return { hit: false, label: "No “not X but Y”" };
  }

  // Match on the raw text, merge overlapping spans, then escape each segment,
  // so overlapping patterns can never nest or split the highlight markup.
  function highlightConstruction(text) {
    const merged = [];
    for (const [s, e] of constructionSpans(text)) {
      const last = merged[merged.length - 1];
      if (last && s < last[1]) last[1] = Math.max(last[1], e);
      else merged.push([s, e]);
    }
    let out = "", pos = 0;
    for (const [s, e] of merged) {
      out += esc(text.slice(pos, s)) + `<span class="construction">${esc(text.slice(s, e))}</span>`;
      pos = e;
    }
    return out + esc(text.slice(pos));
  }

  // ─── playback (side-by-side, lockstep) ──────────────────────────────────
  async function play() {
    if (!EXAMPLES) return;
    const ex = EXAMPLES.examples[currentExampleIdx];
    const rightRun = ex.runs.off;
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

  // ─── wiring ─────────────────────────────────────────────────────────────
  function wireControls() {
    $("play").addEventListener("click", play);
  }

  init().catch(e => {
    console.error("init error:", e);
    setStatus("down", "init error: " + e.message);
  });
})();
