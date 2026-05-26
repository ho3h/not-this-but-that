// Playground v3 — friendlier UX, pan/zoom map, community navigator, concept
// chips. Same daemon contract as before. Two paths for non-technical users:
// presets ("silence the AI-ism coalition") and concepts ("silence everything
// labeled 'negation'"). Power users still get click-to-toggle on the map.

(function () {
  "use strict";

  const PROBE = "/probe";
  const LAYOUT_URL = "/demo/layout.json";
  const COMM_URL = "/demo/community_names.json";

  let LAYOUT = null;
  let COMM = null;
  let CANVAS, CTX, DPR = window.devicePixelRatio || 1;
  let ACTIVITY = null;
  let ABLATED = new Set();
  let MATCHED = new Set();
  let playSession = 0;

  // Pan/zoom view state
  let viewScale = 1, viewX = 0, viewY = 0;
  let dragging = false, dragStartX = 0, dragStartY = 0, dragStartViewX = 0, dragStartViewY = 0;
  let dragMoved = false;

  // Lasso state — shift-drag draws a polygon, on release we add every
  // feature inside it to the ablation set
  let lassoActive = false;
  let lassoPoints = []; // [[x, y] in canvas px], appended every mousemove

  // Highlight a community by id (set when hovering a region in the sidebar)
  let hoveredCid = null;

  // The feature most recently alt-clicked — gets a ring + pulse halo
  let neighbourAnchor = null;
  let neighbourPulse = 0; // 0→1 decaying each frame

  // Demo 3 audit: track the provenance of every silenced feature.
  // Map<feature_idx, Array<{kind, label, ts}>>. Sources accumulate so a feature
  // selected by multiple actions (e.g. preset + manual click) records both.
  const ABLATION_SOURCES = new Map();
  // Session id (random, used as the audit-root)
  const SESSION_ID = "s_" + Math.random().toString(36).slice(2, 10);
  // Last generation's intervention_id, if any
  let LAST_INTERVENTION_ID = null;

  function recordSource(feature_indices, kind, label) {
    const ts = Date.now();
    for (const idx of feature_indices) {
      const list = ABLATION_SOURCES.get(idx) || [];
      list.push({ kind, label, ts });
      ABLATION_SOURCES.set(idx, list);
    }
  }
  function clearSources(feature_indices) {
    if (feature_indices === null) {
      ABLATION_SOURCES.clear();
    } else {
      for (const idx of feature_indices) ABLATION_SOURCES.delete(idx);
    }
  }

  // Preset sets fetched from daemon
  let TOP25 = null, SUPPRESSORS10 = null;
  const MINIMAL = [3223, 9909];

  function $(id) { return document.getElementById(id); }
  function setStatus(state, msg) {
    const el = $("status");
    el.className = state === "up" ? "pill-up" : "pill-down";
    el.textContent = msg;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  // Visual construction detector (shared with story view).
  const PATTERNS = [
    /\b(is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t|don)\s+(?:not\s+)?(?:just\s+)?[^.,;:!?\n]{1,80}[,;—–\-]\s*(?:it'?s?|they'?re?|they|he'?s?|she'?s?|we'?re?|but\s+|but\b)/gi,
    /\b(?:is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t)\s+(?:not|just)\s+(?:just\s+)?[^.!?\n]{1,80}[.!?]\s*(?:It'?s?|They'?re?|He'?s?|She'?s?|We'?re?|But\s+|Rather|Instead)/g,
    /(?:\bless\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*more\b|\bnot\s+about\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*(?:it'?s?\s+about|about))/gi,
  ];
  function hasConstruction(text) {
    for (const p of PATTERNS) { p.lastIndex = 0; if (p.test(text)) return true; }
    return false;
  }
  function highlightConstruction(text) {
    let out = esc(text);
    for (const p of PATTERNS) {
      p.lastIndex = 0;
      out = out.replace(p, m => `<span class="construction">${m}</span>`);
    }
    return out;
  }

  // ─── init ───────────────────────────────────────────────────────────────
  async function init() {
    try {
      [LAYOUT, COMM] = await Promise.all([
        fetch(LAYOUT_URL).then(r => r.json()),
        fetch(COMM_URL).then(r => r.json()).catch(() => ({})),
      ]);
      ACTIVITY = new Float32Array(LAYOUT.features.length);
    } catch (e) {
      $("status").textContent = "FATAL: layout failed — " + e.message;
      return;
    }
    setupCanvas();
    renderCommunityNav();
    wireControls();
    redraw();

    // Daemon health
    try {
      const r = await fetch(PROBE, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "ping" }),
      }).then(r => r.json());
      if (r.ok) setStatus("up", `daemon UP · Gemma 2 2B on ${r.result.device}`);
      else { setStatus("down", "daemon: " + r.error); $("generate").disabled = true; }
    } catch (e) {
      setStatus("down", "daemon down — start with scripts/probe_run.sh start");
      $("generate").disabled = true;
    }

    // Fetch preset sets in the background
    try {
      const r1 = await fetch(PROBE, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "attribution", top_n: 25, kind: "promote" }) }).then(r => r.json());
      if (r1.ok) TOP25 = r1.result.features;
    } catch (e) {}
    try {
      const r2 = await fetch(PROBE, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "attribution", top_n: 10, kind: "suppress" }) }).then(r => r.json());
      if (r2.ok) SUPPRESSORS10 = r2.result.features;
    } catch (e) {}
  }

  // ─── canvas + cloud ─────────────────────────────────────────────────────
  function setupCanvas() {
    CANVAS = $("canvas");
    CTX = CANVAS.getContext("2d");
    const resize = () => {
      const r = CANVAS.getBoundingClientRect();
      CANVAS.width = r.width * DPR;
      CANVAS.height = r.height * DPR;
      redraw();
    };
    window.addEventListener("resize", resize);
    resize();
    CANVAS.addEventListener("mousemove", onCanvasMove);
    CANVAS.addEventListener("mousedown", onCanvasDown);
    CANVAS.addEventListener("mouseup", onCanvasUp);
    CANVAS.addEventListener("mouseleave", () => {
      $("tip").style.display = "none";
      dragging = false; CANVAS.style.cursor = "crosshair";
    });
    CANVAS.addEventListener("wheel", onWheel, { passive: false });
    $("zoom-in").addEventListener("click", () => zoomAt(1.4, CANVAS.width/2, CANVAS.height/2));
    $("zoom-out").addEventListener("click", () => zoomAt(1/1.4, CANVAS.width/2, CANVAS.height/2));
    $("zoom-reset").addEventListener("click", () => { viewScale = 1; viewX = 0; viewY = 0; redraw(); });
  }

  // World coords are layout's [-1,1] space. xy() maps to canvas pixels using
  // the current pan/zoom view state.
  function xy(f) {
    const W = CANVAS.width, H = CANVAS.height;
    const m = Math.min(W, H) * 0.46 * viewScale;
    return [W / 2 + viewX + f.x * m, H / 2 + viewY + f.y * m];
  }
  function zoomAt(factor, cx, cy) {
    // Keep the world point under (cx,cy) fixed while scaling
    const W = CANVAS.width, H = CANVAS.height;
    const mBefore = Math.min(W, H) * 0.46 * viewScale;
    const worldX = (cx - W/2 - viewX) / mBefore;
    const worldY = (cy - H/2 - viewY) / mBefore;
    viewScale = Math.max(0.5, Math.min(viewScale * factor, 12));
    const mAfter = Math.min(W, H) * 0.46 * viewScale;
    viewX = cx - W/2 - worldX * mAfter;
    viewY = cy - H/2 - worldY * mAfter;
    redraw();
  }
  function onWheel(ev) {
    ev.preventDefault();
    const r = CANVAS.getBoundingClientRect();
    const cx = (ev.clientX - r.left) * DPR;
    const cy = (ev.clientY - r.top) * DPR;
    const factor = ev.deltaY < 0 ? 1.12 : 1/1.12;
    zoomAt(factor, cx, cy);
  }

  const COMMUNITY_HUE = {};
  function commColor(cid, alpha = 0.5) {
    if (cid === 12) return `rgba(198, 145, 22, ${alpha})`;  // 3223's community in mustard
    if (!(cid in COMMUNITY_HUE)) {
      const k = Object.keys(COMMUNITY_HUE).length;
      COMMUNITY_HUE[cid] = (k * 137.508) % 360;
    }
    return `hsla(${COMMUNITY_HUE[cid]}, 35%, 50%, ${alpha})`;
  }

  function redraw() {
    if (!LAYOUT || !CANVAS) return;
    const W = CANVAS.width, H = CANVAS.height;
    CTX.clearRect(0, 0, W, H);
    if (ACTIVITY) for (let i = 0; i < ACTIVITY.length; i++) ACTIVITY[i] *= 0.93;

    // Underlay: community regions as translucent blobs
    drawRegions(W, H);

    const feats = LAYOUT.features;
    const rBase = (0.85 + Math.sqrt(viewScale) * 0.3) * DPR;
    for (let i = 0; i < feats.length; i++) {
      const f = feats[i];
      const [x, y] = xy(f);
      if (x < -10 || x > W + 10 || y < -10 || y > H + 10) continue;
      const act = ACTIVITY ? ACTIVITY[i] : 0;
      let color, r;
      if (ABLATED.has(i)) {
        color = "rgba(183, 60, 42, 0.95)"; r = rBase * 2.4;
      } else if (act > 0.7) {
        color = "rgba(198, 145, 22, 0.95)"; r = rBase * 3.0;
      } else if (act > 0.05) {
        const a = 0.4 + 0.6 * act;
        color = `rgba(44, 138, 74, ${a.toFixed(2)})`;
        r = rBase * (1.4 + 1.2 * act);
      } else if (MATCHED.has(i)) {
        color = "rgba(74, 111, 165, 0.95)"; r = rBase * 3.2;
      } else {
        color = commColor(f.cid, 0.18);
        r = rBase * 0.9;
      }
      CTX.fillStyle = color;
      CTX.beginPath();
      CTX.arc(x, y, r, 0, Math.PI * 2);
      CTX.fill();
    }
    // halos
    if (ACTIVITY) {
      for (let i = 0; i < ACTIVITY.length; i++) {
        if (ACTIVITY[i] > 0.7 && !ABLATED.has(i)) {
          const [x, y] = xy(feats[i]);
          CTX.strokeStyle = "rgba(198, 145, 22, 0.55)";
          CTX.lineWidth = 1.3 * DPR;
          CTX.beginPath();
          CTX.arc(x, y, 7 * DPR, 0, Math.PI * 2);
          CTX.stroke();
        }
      }
    }
    for (const i of ABLATED) {
      const [x, y] = xy(feats[i]);
      if (x < -10 || x > W + 10 || y < -10 || y > H + 10) continue;
      CTX.strokeStyle = "rgba(183, 60, 42, 0.4)";
      CTX.lineWidth = 1 * DPR;
      CTX.beginPath();
      CTX.arc(x, y, 5 * DPR, 0, Math.PI * 2);
      CTX.stroke();
    }
    // Bright ring around every MATCHED (search-result / alt-click neighbour) feature
    if (MATCHED.size > 0) {
      CTX.strokeStyle = "rgba(74, 111, 165, 0.7)";
      CTX.lineWidth = 1.6 * DPR;
      for (const i of MATCHED) {
        if (ABLATED.has(i)) continue;
        const [x, y] = xy(feats[i]);
        if (x < -10 || x > W + 10 || y < -10 || y > H + 10) continue;
        CTX.beginPath();
        CTX.arc(x, y, 6 * DPR, 0, Math.PI * 2);
        CTX.stroke();
      }
    }
    // Big saturated halo on the alt-click anchor + pulse
    if (neighbourAnchor !== null && LAYOUT.features[neighbourAnchor]) {
      const [x, y] = xy(LAYOUT.features[neighbourAnchor]);
      const r = (8 + 6 * neighbourPulse) * DPR;
      CTX.strokeStyle = `rgba(74, 111, 165, ${0.4 + 0.5 * neighbourPulse})`;
      CTX.lineWidth = 2 * DPR;
      CTX.beginPath();
      CTX.arc(x, y, r, 0, Math.PI * 2);
      CTX.stroke();
      // Inner solid dot
      CTX.fillStyle = "rgba(74, 111, 165, 0.95)";
      CTX.beginPath();
      CTX.arc(x, y, 3.5 * DPR, 0, Math.PI * 2);
      CTX.fill();
      neighbourPulse = Math.max(0, neighbourPulse - 0.04);
      if (neighbourPulse > 0) startAnim();
    }
    if (lassoActive && lassoPoints.length >= 2) drawLasso();
  }

  function drawLasso() {
    CTX.strokeStyle = "rgba(196, 106, 31, 0.85)";
    CTX.lineWidth = 1.4 * DPR;
    CTX.fillStyle = "rgba(196, 106, 31, 0.10)";
    CTX.beginPath();
    CTX.moveTo(lassoPoints[0][0], lassoPoints[0][1]);
    for (let i = 1; i < lassoPoints.length; i++) {
      CTX.lineTo(lassoPoints[i][0], lassoPoints[i][1]);
    }
    CTX.closePath();
    CTX.fill();
    CTX.stroke();
    // Live count of features inside
    if (lassoPoints.length > 8) {
      const inside = featuresInPolygon(lassoPoints).length;
      const [cx, cy] = lassoPoints[lassoPoints.length - 1];
      CTX.font = `600 ${12 * DPR}px -apple-system, system-ui, sans-serif`;
      CTX.fillStyle = "rgba(196, 106, 31, 0.95)";
      CTX.textAlign = "left";
      CTX.textBaseline = "bottom";
      CTX.fillText(`${inside} features`, cx + 8*DPR, cy - 4*DPR);
    }
  }

  // ─── Region rendering (translucent coloured blobs behind dots) ─────────
  // Computes centroid + extent per community in world (UMAP) coords. Cached.
  let COMM_STATS = null;
  function computeCommunityStats() {
    if (COMM_STATS) return COMM_STATS;
    const byCid = {};
    for (const f of LAYOUT.features) {
      const slot = byCid[f.cid] || (byCid[f.cid] = { xs: [], ys: [], cid: f.cid });
      slot.xs.push(f.x); slot.ys.push(f.y);
    }
    const out = {};
    for (const cid in byCid) {
      const s = byCid[cid];
      const n = s.xs.length;
      const cx = s.xs.reduce((a, b) => a + b, 0) / n;
      const cy = s.ys.reduce((a, b) => a + b, 0) / n;
      const vx = s.xs.reduce((a, b) => a + (b - cx) ** 2, 0) / n;
      const vy = s.ys.reduce((a, b) => a + (b - cy) ** 2, 0) / n;
      out[cid] = {
        cid: +cid, n, cx, cy,
        rx: Math.max(Math.sqrt(vx) * 1.4, 0.05),
        ry: Math.max(Math.sqrt(vy) * 1.4, 0.05),
        name: (COMM && COMM[cid]) ? COMM[cid].name : `community ${cid}`,
      };
    }
    COMM_STATS = out;
    return out;
  }

  function drawRegions(W, H) {
    if (!COMM) return;
    const stats = computeCommunityStats();
    const items = Object.values(stats).sort((a, b) => b.n - a.n);
    const m = Math.min(W, H) * 0.46 * viewScale;
    for (const s of items) {
      const cx = W / 2 + viewX + s.cx * m;
      const cy = H / 2 + viewY + s.cy * m;
      const rx = Math.max(s.rx * m, 24 * DPR);
      const ry = Math.max(s.ry * m, 24 * DPR);
      const rMax = Math.max(rx, ry);
      // Cull off-screen
      if (cx + rMax < 0 || cx - rMax > W || cy + rMax < 0 || cy - rMax > H) continue;

      const hovered = (hoveredCid === s.cid);
      const alphaInner = hovered ? 0.32 : 0.13;
      const alphaMid   = hovered ? 0.10 : 0.045;

      const grad = CTX.createRadialGradient(cx, cy, 0, cx, cy, rMax);
      grad.addColorStop(0,    commColor(s.cid, alphaInner));
      grad.addColorStop(0.65, commColor(s.cid, alphaMid));
      grad.addColorStop(1,    commColor(s.cid, 0));

      CTX.save();
      CTX.translate(cx, cy);
      CTX.scale(rx / rMax, ry / rMax);
      CTX.fillStyle = grad;
      CTX.beginPath();
      CTX.arc(0, 0, rMax, 0, Math.PI * 2);
      CTX.fill();
      CTX.restore();

      // Hovered region: also draw a faint dashed outline so the user is sure
      if (hovered) {
        CTX.save();
        CTX.strokeStyle = commColor(s.cid, 0.55);
        CTX.setLineDash([6 * DPR, 4 * DPR]);
        CTX.lineWidth = 1.4 * DPR;
        CTX.translate(cx, cy);
        CTX.scale(rx / rMax, ry / rMax);
        CTX.beginPath();
        CTX.arc(0, 0, rMax * 0.95, 0, Math.PI * 2);
        CTX.stroke();
        CTX.restore();
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

  function nearestFeature(ev) {
    if (!LAYOUT) return null;
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
    const r = 14 * DPR;
    return nd < r * r ? nearest : null;
  }

  function onCanvasDown(ev) {
    if (ev.shiftKey) {
      // Shift-drag: lasso selection
      lassoActive = true;
      const rect = CANVAS.getBoundingClientRect();
      lassoPoints = [[(ev.clientX - rect.left) * DPR, (ev.clientY - rect.top) * DPR]];
      CANVAS.style.cursor = "crosshair";
      $("tip").style.display = "none";
      redraw();
      return;
    }
    dragging = true; dragMoved = false;
    dragStartX = ev.clientX; dragStartY = ev.clientY;
    dragStartViewX = viewX; dragStartViewY = viewY;
    CANVAS.style.cursor = "grabbing";
  }
  function onCanvasUp(ev) {
    if (lassoActive) {
      // Finalize: add every feature inside the polygon to the ablation set
      lassoActive = false;
      if (lassoPoints.length >= 3) {
        const inside = featuresInPolygon(lassoPoints);
        if (inside.length > 0) {
          addToAblation(inside, { kind: "lasso",
            label: `lasso (${inside.length} features inside polygon)` });
        }
      }
      lassoPoints = [];
      CANVAS.style.cursor = "crosshair";
      redraw();
      return;
    }
    if (dragging && !dragMoved) {
      const i = nearestFeature(ev);
      if (i !== null) {
        if (ev.altKey || ev.metaKey) {
          showNeighbours(i);
        } else {
          toggleAblation(i);
        }
      }
    }
    dragging = false; CANVAS.style.cursor = "crosshair";
  }
  function onCanvasMove(ev) {
    if (lassoActive) {
      const rect = CANVAS.getBoundingClientRect();
      lassoPoints.push([(ev.clientX - rect.left) * DPR, (ev.clientY - rect.top) * DPR]);
      redraw();
      return;
    }
    if (dragging) {
      const dx = (ev.clientX - dragStartX) * DPR;
      const dy = (ev.clientY - dragStartY) * DPR;
      if (dx*dx + dy*dy > 9) dragMoved = true;
      viewX = dragStartViewX + dx;
      viewY = dragStartViewY + dy;
      redraw();
      return;
    }
    const i = nearestFeature(ev);
    const tip = $("tip");
    if (i === null) { tip.style.display = "none"; return; }
    const f = LAYOUT.features[i];
    const commName = COMM && COMM[f.cid] ? COMM[f.cid].name : `community ${f.cid}`;
    const rect = CANVAS.getBoundingClientRect();
    tip.style.display = "block";
    tip.style.left = Math.min(ev.clientX - rect.left + 14, rect.width - 320) + "px";
    tip.style.top = Math.min(ev.clientY - rect.top + 10, rect.height - 80) + "px";
    tip.innerHTML = `<span class="idx">feature ${f.idx}</span>
      <span class="lbl">${esc(f.label) || "(no auto-interp label)"}</span>
      <div class="meta">in: ${esc(commName)} · density ${(f.density*100).toFixed(1)}%<br>
      <kbd>click</kbd> to ${ABLATED.has(f.idx) ? "un-" : ""}silence ·
      <kbd>⌥ click</kbd> for graph neighbours ·
      <kbd>⇧ drag</kbd> to lasso</div>`;
  }

  function pointInPolygon(px, py, poly) {
    // ray casting
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1];
      const xj = poly[j][0], yj = poly[j][1];
      const intersect = ((yi > py) !== (yj > py)) &&
                        (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function featuresInPolygon(poly) {
    const hits = [];
    const feats = LAYOUT.features;
    for (let i = 0; i < feats.length; i++) {
      const [x, y] = xy(feats[i]);
      if (pointInPolygon(x, y, poly)) hits.push(i);
    }
    return hits;
  }

  // "What concepts does this prompt touch?" — RAG-for-activations.
  // Embeds the prompt server-side, vector-searches the 16k feature labels,
  // highlights matches on the cloud, surfaces them in search-info. User
  // decides whether to silence by clicking "Silence matches".
  async function retrieveConcepts() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return;
    const btn = $("retrieve-btn");
    btn.disabled = true;
    const orig = btn.innerHTML;
    btn.innerHTML = "◆ Asking Neo4j…";
    $("search").value = "";
    $("search-info").innerHTML = `<span style="color: var(--neo);">◆</span> Embedding prompt and vector-searching 16,384 feature labels…`;
    try {
      const r = await fetch(PROBE, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cmd: "concept_retrieve", prompt, k: 15, expand_neighbours: 0,
        }),
      }).then(r => r.json());
      if (!r.ok) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">error: ${esc(r.error || "unknown")}</span>`;
        return;
      }
      const res = r.result;
      MATCHED = new Set(res.features);
      neighbourAnchor = null; neighbourPulse = 0;
      const top = res.matches.slice(0, 5).map(m =>
        `<span style="display: inline-block; background: var(--neo-soft); color: var(--neo); padding: 1px 6px; border-radius: 3px; margin: 2px 3px 0 0; font-size: 11px;">${esc(m.label.slice(0,42))}</span>`
      ).join("");
      $("search-info").innerHTML = `<strong>${res.matches.length}</strong> features semantically match this prompt <span class="neo-tag">◆ Vector search</span><div style="margin-top: 4px;">${top}</div><div style="font-size: 11px; color: var(--faint); margin-top: 4px;">These are what the model is "thinking about." Click <em>Silence matches</em> to constrain it — see how it routes around the missing concepts.</div>`;
      $("search").value = `concepts in: ${prompt.slice(0, 30)}…`;
      startAnim();
      setTimeout(stopAnim, 1500);
    } catch (e) {
      $("search-info").innerHTML = `<span style="color: var(--oppose);">network: ${esc(e.message)}</span>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = orig;
    }
  }

  // ── Demo 2: Mixer (compose multiple behaviours at chosen intensities) ──
  let MIXER_LAST_PAYLOAD = null;
  function readMixerIntensities() {
    const out = {};
    document.querySelectorAll(".pg-slider input[type='range']").forEach(s => {
      const v = parseInt(s.value, 10);
      if (v > 0) out[s.dataset.name] = v;
    });
    return out;
  }
  function renderMixerInfo() {
    const intensities = readMixerIntensities();
    const info = $("mixer-info");
    const names = Object.keys(intensities);
    if (names.length === 0) {
      info.textContent = "All sliders at 0 — no behaviours selected.";
      return;
    }
    info.innerHTML = `<strong>${names.length} behaviour${names.length === 1 ? '' : 's'} engaged:</strong> ` +
      names.map(n => `${n} @ ${intensities[n]}%`).join(" · ");
  }
  async function applyMixer() {
    const intensities = readMixerIntensities();
    if (Object.keys(intensities).length === 0) {
      $("mixer-info").innerHTML = `<span style="color: var(--oppose);">Move at least one slider above 0.</span>`;
      return;
    }
    const btn = $("mixer-apply");
    btn.disabled = true; const orig = btn.textContent;
    btn.textContent = "◆ Composing via Cypher…";
    try {
      const r = await fetch(PROBE, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "compose_behaviours", intensities }),
      }).then(r => r.json());
      if (!r.ok) {
        $("mixer-info").innerHTML = `<span style="color: var(--oppose);">error: ${esc(r.error || "unknown")}</span>`;
        return;
      }
      const res = r.result;
      MIXER_LAST_PAYLOAD = res;
      // Replace ABLATED with the composed set, tagged per behaviour for the audit
      ABLATED = new Set(res.features);
      ABLATION_SOURCES.clear();
      for (const b of res.per_behaviour) {
        recordSource(b.features, "mixer",
          `mixer: ${b.name} @${b.intensity}% (took ${b.took}/${b.of})`);
      }
      renderAblated();
      // Cypher display
      $("mixer-cypher").textContent = res.cypher;
      // Summary
      const parts = res.per_behaviour.map(b =>
        `${b.name} @${b.intensity}% → ${b.took} feats`
      ).join(" · ");
      $("mixer-info").innerHTML = `<strong>${res.n_total} features composed</strong> · ${parts} <span class="neo-tag">◆ Cypher UNION</span>`;
      redraw();
      // Now auto-generate
      btn.textContent = "▶ Generating…";
      await generate();
    } catch (e) {
      $("mixer-info").innerHTML = `<span style="color: var(--oppose);">network: ${esc(e.message)}</span>`;
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
    }
  }

  // ── Demo 1: Surgical de-slop (concept retrieval ∩ behaviour coalition) ──
  // Asks the daemon to intersect prompt-relevant concepts with the AI-ism
  // coalition stored as a :Behaviour node in Neo4j. Silences only the
  // overlap — i.e. only the AI-ism features the prompt would have used.
  // This is graph set algebra in product form.
  async function surgicalDeslop() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return;
    const btn = $("surgical-btn");
    btn.disabled = true;
    const orig = btn.innerHTML;
    btn.innerHTML = "◆ Cypher running…";
    $("search").value = "";
    $("search-info").innerHTML = `<span style="color: var(--neo);">◆</span> Retrieving concepts, intersecting with AI-ism coalition…`;
    try {
      const r = await fetch(PROBE, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cmd: "surgical_deslop", prompt, behaviour: "ai-ism", k: 80,
        }),
      }).then(r => r.json());
      if (!r.ok) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">error: ${esc(r.error || "unknown")}</span>`;
        return;
      }
      const res = r.result;
      MATCHED = new Set(res.retrieved_features);
      neighbourAnchor = null; neighbourPulse = 0;

      // The intersection becomes the new ablation set, tagged with both
      // sources (concept retrieval + behaviour membership) for the audit
      if (res.intersection_features.length > 0) {
        ABLATED = new Set(res.intersection_features);
        ABLATION_SOURCES.clear();
        recordSource(res.intersection_features, "surgical_deslop",
          `surgical: ${prompt.slice(0, 32)}… ∩ AI-ism coalition`);
        renderAblated();
      }

      // Pretty UI: show both the matches AND the Cypher
      const matchChips = res.matches.map(m =>
        `<span style="display: inline-block; background: rgba(183,60,42,0.12); color: var(--oppose); padding: 1px 7px; border-radius: 3px; margin: 2px 3px 0 0; font-size: 11px;">feat ${m.idx} · ${esc(m.label.slice(0,32))}</span>`
      ).join("");

      if (res.intersection_features.length === 0) {
        // Build the ranked-fallback chip list (top 5 coalition features by per-prompt similarity)
        const ranked = (res.ranked_coalition || []).slice(0, 5);
        const rankedChips = ranked.map(m =>
          `<span style="display: inline-block; background: rgba(60,107,182,0.12); color: var(--neo); padding: 1px 7px; border-radius: 3px; margin: 2px 3px 0 0; font-size: 11px;">feat ${m.idx} · sim ${m.score.toFixed(2)} · ${esc(m.label.slice(0,32))}</span>`
        ).join("");
        $("search-info").innerHTML = `
          <strong>✓ No strict intersection.</strong>
          <span style="color: var(--faint);">Of ${res.n_retrieved} concepts the prompt touches, <strong>0</strong> are in the top-K of the AI-ism coalition.</span>
          <div style="margin-top: 6px; font-size: 11px; color: var(--faint);">This prompt doesn't sit in the topical territory where the chatbot's slop typically lives. The graph said "leave it alone."</div>
          <div style="margin-top: 8px; padding-top: 6px; border-top: 1px solid var(--rule);">
            <span style="font-size: 11px; color: var(--faint);">Or, ranked by per-prompt relevance, the 5 most-related coalition features:</span>
            <div style="margin-top: 4px;">${rankedChips}</div>
            <button class="btn-chip" id="surgical-rank-apply" style="margin-top: 6px; font-size: 11px;">Silence the top 5 ranked</button>
          </div>
          <details style="margin-top: 8px; font-size: 11px;">
            <summary style="color: var(--neo); cursor: pointer;">show Cypher</summary>
            <pre style="margin: 4px 0 0; font-family: var(--mono); font-size: 11px; color: var(--muted); background: var(--bg); padding: 6px 8px; border-radius: 3px; white-space: pre-wrap;">${esc(res.cypher)}</pre>
          </details>`;
        // Wire the rank-apply button to silence the top-5 by relevance
        const rankBtn = document.getElementById("surgical-rank-apply");
        if (rankBtn) {
          rankBtn.addEventListener("click", () => {
            const topIds = ranked.map(m => m.idx);
            ABLATED = new Set(topIds);
            ABLATION_SOURCES.clear();
            recordSource(topIds, "surgical_ranked",
              `top-5 of AI-ism ranked by ${prompt.slice(0, 32)}…`);
            renderAblated();
            $("search-info").innerHTML += `<div style="margin-top: 4px; color: var(--oppose); font-size: 11px;">✓ ${topIds.length} top-ranked coalition features silenced. Hit ▶ Generate.</div>`;
          }, { once: true });
        }
      } else {
        $("search-info").innerHTML = `
          <strong>${res.n_intersection}</strong> AI-ism features silenced <span style="color: var(--faint);">— the overlap of ${res.n_retrieved} prompt-concepts ∩ ${res.n_behaviour} coalition features</span> <span class="neo-tag">◆ Cypher intersection</span>
          <div style="margin-top: 4px;">${matchChips}</div>
          <div style="margin-top: 6px; font-size: 11px; color: var(--faint);">These are the slop features your prompt would have invoked. Click <strong>▶ Generate</strong> to see the difference.</div>
          <details style="margin-top: 8px; font-size: 11px;">
            <summary style="color: var(--neo); cursor: pointer;">show Cypher</summary>
            <pre style="margin: 4px 0 0; font-family: var(--mono); font-size: 11px; color: var(--muted); background: var(--bg); padding: 6px 8px; border-radius: 3px; white-space: pre-wrap;">${esc(res.cypher)}</pre>
          </details>`;
      }
      $("search").value = `surgical: ${prompt.slice(0, 30)}…`;
      startAnim();
      setTimeout(stopAnim, 1500);
    } catch (e) {
      $("search-info").innerHTML = `<span style="color: var(--oppose);">network: ${esc(e.message)}</span>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = orig;
    }
  }

  // Alt-click: show this feature's top-10 decoder-cosine neighbours from Neo4j.
  // Immediate visual feedback (pulse on anchor) before the response lands; then
  // highlight neighbours BIG and BLUE so it's unmistakable.
  async function showNeighbours(idx) {
    // 1. Immediate feedback — pulse the clicked feature so the user sees the gesture registered
    neighbourAnchor = idx;
    neighbourPulse = 1.0;
    const lbl = LAYOUT.features[idx].label || "(no label)";
    $("search-info").innerHTML = `<span style="color: var(--neo);">◆</span> Asking Neo4j for the decoder-cosine neighbours of feat ${idx}…`;
    $("search").value = `neighbours of ${idx}`;
    startAnim();
    try {
      const r = await fetch(PROBE, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "graph", query: "decoder_neighbors",
                                anchor: idx, k: 10 }),
      }).then(r => r.json());
      if (!r.ok) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">error fetching neighbours: ${esc(r.error || "unknown")}</span>`;
        return;
      }
      const feats = r.result.features;
      MATCHED = new Set(feats);
      MATCHED.add(idx);
      neighbourPulse = 1.0;  // re-pulse on data arrival
      $("search-info").innerHTML = `<strong>${feats.length}</strong> decoder-cosine neighbours of feat ${idx} (<em>${esc(lbl).slice(0,40)}</em>) — highlighted in blue. <span class="neo-tag">◆ Cypher</span><div style="margin-top: 4px; font-size: 11px; color: var(--faint);">Click "Silence all matches" to ablate this cluster.</div>`;
      startAnim();
      setTimeout(stopAnim, 1500);
    } catch (e) {
      $("search-info").innerHTML = `<span style="color: var(--oppose);">network error: ${esc(e.message)}</span>`;
    }
  }

  // ─── ablation set ──────────────────────────────────────────────────────
  // Every mutation accepts a source ({kind, label}) so the audit trail can
  // explain why each feature was silenced.
  function toggleAblation(i, source = { kind: "user_click", label: "manual click on map" }) {
    if (ABLATED.has(i)) {
      ABLATED.delete(i);
      clearSources([i]);
    } else {
      ABLATED.add(i);
      recordSource([i], source.kind, source.label);
    }
    renderAblated();
    redraw();
  }
  function setAblation(arr, source = { kind: "preset", label: "preset" }) {
    ABLATED = new Set(arr);
    ABLATION_SOURCES.clear();
    recordSource(arr, source.kind, source.label);
    renderAblated();
    redraw();
  }
  function addToAblation(arr, source = { kind: "added", label: "added" }) {
    const newly = [];
    for (const i of arr) {
      if (!ABLATED.has(i)) newly.push(i);
      ABLATED.add(i);
    }
    recordSource(arr, source.kind, source.label);  // record on all (overlaps too)
    renderAblated();
    redraw();
  }
  function clearAblation() {
    ABLATED = new Set();
    ABLATION_SOURCES.clear();
    document.querySelectorAll(".preset-card").forEach(c => c.classList.toggle("on", c.dataset.preset === "none"));
    renderAblated();
    redraw();
  }
  function renderAblated() {
    $("ablated-count").textContent = ABLATED.size;
    const list = $("ablated-list");
    if (ABLATED.size === 0) {
      list.innerHTML = `<div style="font-family: var(--sans); font-size: 11px; color: var(--faint);">Pick a preset, type a concept, click a community, or click features on the map.</div>`;
      return;
    }
    const items = [...ABLATED].sort((a, b) => a - b);
    list.innerHTML = items.map(i => {
      const f = LAYOUT.features[i];
      const lbl = (f.label || "(no label)").slice(0, 32);
      return `<div class="feat-row ablated" data-idx="${i}">
        <span class="idx">${i}</span>
        <span class="lbl">${esc(lbl)}</span>
        <span class="x" data-rm="${i}">×</span>
      </div>`;
    }).join("");
    list.querySelectorAll("[data-rm]").forEach(el => {
      el.addEventListener("click", ev => {
        ev.stopPropagation();
        toggleAblation(parseInt(el.dataset.rm, 10));
      });
    });
  }

  // ─── search by label ───────────────────────────────────────────────────
  function runSearch(q) {
    MATCHED.clear();
    if (!q) {
      $("search-info").textContent = "Type to filter features by their label.";
      redraw(); return;
    }
    const needle = q.toLowerCase();
    let n = 0;
    for (const f of LAYOUT.features) {
      if (f.label && f.label.toLowerCase().includes(needle)) {
        MATCHED.add(f.idx); n++;
      }
    }
    $("search-info").innerHTML = n === 0
      ? `<span style="color: var(--oppose);">no matches for "${esc(q)}"</span>`
      : `<strong style="color: var(--ink);">${n}</strong> feature${n===1?"":"s"} match "${esc(q)}" — highlighted blue on the map`;
    redraw();
  }

  // ─── presets ───────────────────────────────────────────────────────────
  function pickPreset(name) {
    document.querySelectorAll(".preset-card").forEach(c => c.classList.toggle("on", c.dataset.preset === name));
    if (name === "none") return clearAblation();
    if (name === "top25")        return setAblation(TOP25 || [],         { kind: "preset", label: "preset: AI-ism coalition (top-25)" });
    if (name === "minimal")      return setAblation(MINIMAL,             { kind: "preset", label: "preset: minimal core (3223+9909)" });
    if (name === "suppressors")  return setAblation(SUPPRESSORS10 || [], { kind: "preset", label: "preset: suppressors top-10" });
  }

  // ─── community navigator ───────────────────────────────────────────────
  // Lists communities sorted by size. Each row hover → highlights the region
  // on the map; click → silences its top features.
  function renderCommunityNav() {
    const nav = $("comm-nav");
    if (!COMM) { nav.innerHTML = "<div style='color: var(--faint); font-size: 11px;'>(community names not loaded)</div>"; return; }
    const entries = Object.values(COMM).sort((a, b) => b.size - a.size);
    nav.innerHTML = entries.map(e => {
      // Build a colour swatch using the same hue function as the map
      // We can't call commColor before the canvas is ready; use a CSS
      // variable that the redraw will style later — but it's simpler to
      // just compute the inline style with the same golden-angle hue logic.
      const swatchHue = (e.cid === 12) ? 41 : ((Object.keys(COMM).indexOf(String(e.cid))) * 137.508) % 360;
      const sat = (e.cid === 12) ? "70%" : "45%";
      return `
      <div class="comm-item" data-cid="${e.cid}" title="${esc(e.exemplar_labels.slice(0,3).join(' · '))}">
        <span class="swatch" style="background: hsla(${swatchHue}, ${sat}, 55%, 0.55)"></span>
        <span class="nm">${esc(e.name)}</span>
        <span class="sz">${e.size}</span>
      </div>`;
    }).join("");
    nav.querySelectorAll(".comm-item").forEach(item => {
      const cid = parseInt(item.dataset.cid, 10);
      item.addEventListener("mouseenter", () => {
        hoveredCid = cid;
        startAnim();
      });
      item.addEventListener("mouseleave", () => {
        hoveredCid = null;
        startAnim();
        setTimeout(stopAnim, 600);
      });
      item.addEventListener("click", () => {
        const members = LAYOUT.features
          .filter(f => f.cid === cid)
          .sort((a, b) => b.density - a.density)
          .slice(0, 10)
          .map(f => f.idx);
        addToAblation(members, { kind: "community_click",
          label: `community: ${(COMM[cid] && COMM[cid].name) || `cid ${cid}`} (top-10)` });
      });
    });
  }

  // ─── generate ──────────────────────────────────────────────────────────
  async function generate() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return;
    const model = $("model").value;
    const max_new = parseInt($("max_new").value, 10);
    const seed = parseInt($("seed").value, 10);
    const speedMs = parseInt($("speed").value, 10);
    const ablate = [...ABLATED];

    playSession += 1;
    const my = playSession;
    $("generate").disabled = true;
    $("generate").textContent = "▶ Generating…";
    $("text-left").textContent = ""; $("text-right").textContent = "";
    $("vl").textContent = ""; $("vr").textContent = "";
    $("ablated-label").firstChild.textContent = ablate.length
      ? `With ${ablate.length} feature${ablate.length === 1 ? "" : "s"} silenced  `
      : "With nothing silenced (same as baseline)  ";
    ACTIVITY.fill(0);
    startAnim();

    try {
      $("hud_step").textContent = "calling daemon…";
      const t0 = performance.now();
      const [baseR, ablR] = await Promise.all([
        fetch(PROBE, { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cmd: "generate_with_activations", model, prompt,
            ablate: null, max_new_tokens: max_new, seed, top_k_features: 10 }) }).then(r => r.json()),
        ablate.length === 0
          ? Promise.resolve(null)
          : fetch(PROBE, { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ cmd: "generate_with_activations", model, prompt,
                ablate, max_new_tokens: max_new, seed, top_k_features: 10 }) }).then(r => r.json()),
      ]);

      if (my !== playSession) return;
      if (!baseR.ok) throw new Error(baseR.error || "baseline failed");
      if (ablR && !ablR.ok) throw new Error(ablR.error || "ablated failed");

      const baseRecs = baseR.result.records;
      const ablRecs = ablR ? ablR.result.records : baseRecs;
      const maxLen = Math.max(baseRecs.length, ablRecs.length);
      const leftBuf = [], rightBuf = [];

      for (let i = 0; i < maxLen; i++) {
        if (my !== playSession) break;
        $("hud_step").textContent = `step ${i + 1} / ${maxLen}`;
        if (i < baseRecs.length) {
          leftBuf.push(baseRecs[i].token_str);
          $("text-left").innerHTML = highlightConstruction(leftBuf.join(""));
        }
        if (i < ablRecs.length) {
          rightBuf.push(ablRecs[i].token_str);
          $("text-right").innerHTML = highlightConstruction(rightBuf.join(""));
        }
        const rec = ablRecs[Math.min(i, ablRecs.length - 1)];
        for (const f of (rec.top_features || [])) ACTIVITY[f.idx] = 1.0;
        renderActiveList(rec.top_features || []);
        $("active-count").textContent = (rec.top_features || []).length;
        await sleep(speedMs);
      }

      const baseHit = hasConstruction(leftBuf.join(""));
      const ablHit = hasConstruction(rightBuf.join(""));
      $("vl").className = "verdict " + (baseHit ? "hit" : "clean");
      $("vl").textContent = baseHit ? "✓ construction present" : "· clean";
      $("vr").className = "verdict " + (ablHit ? "hit" : "clean");
      $("vr").textContent = ablHit ? "✓ construction present" : "· clean";
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      $("hud_step").textContent = `done · ${maxLen} tokens · ${dt}s`;
      setTimeout(stopAnim, 2000);

      // ── Demo 3: render audit panel + persist to Neo4j ──
      renderAudit(prompt, leftBuf.join(""), rightBuf.join(""));
      persistAudit(prompt, leftBuf.join(""), rightBuf.join(""))
        .catch(e => console.warn("persist failed:", e));
    } catch (e) {
      $("hud_step").textContent = "ERROR: " + e.message;
    } finally {
      $("generate").disabled = false;
      $("generate").textContent = "▶ Generate";
    }
  }
  function renderActiveList(items) {
    const list = $("active-list");
    list.innerHTML = items.map(t => {
      const f = LAYOUT.features[t.idx];
      const lbl = (f.label || "(no label)").slice(0, 32);
      const isAbl = ABLATED.has(t.idx);
      return `<div class="feat-row ${isAbl ? 'ablated' : ''}">
        <span class="idx">${t.idx}</span>
        <span class="lbl">${esc(lbl)}</span>
        <span class="act">${t.act.toFixed(1)}</span>
      </div>`;
    }).join("");
  }
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Demo 3: render the audit panel after a generation ─────────────────
  // Groups silenced features by source kind, shows each feature pill
  // (highlighted gold if multi-source — selected by 2+ different actions).
  function renderAudit(prompt, baseline_text, ablated_text) {
    const panel = $("audit-panel");
    if (ABLATED.size === 0) { panel.style.display = "none"; return; }
    panel.style.display = "block";

    // Group features by source kind
    const byKind = new Map();  // kind → { label, indices: Set<int> }
    let multiSourceFeatures = 0;
    for (const [idx, sources] of ABLATION_SOURCES.entries()) {
      if (!ABLATED.has(idx)) continue;  // skip stale entries
      const uniqueKinds = new Set(sources.map(s => s.kind));
      if (uniqueKinds.size > 1) multiSourceFeatures++;
      for (const s of sources) {
        const key = s.kind + "::" + s.label;
        if (!byKind.has(key)) byKind.set(key, { kind: s.kind, label: s.label, indices: new Set() });
        byKind.get(key).indices.add(idx);
      }
    }

    const groups = [...byKind.values()].sort((a, b) => b.indices.size - a.indices.size);
    const totalMultiSource = multiSourceFeatures;
    const featurePill = (i) => {
      const sources = ABLATION_SOURCES.get(i) || [];
      const kinds = new Set(sources.map(s => s.kind));
      const isMulti = kinds.size > 1;
      const lbl = LAYOUT.features[i] ? (LAYOUT.features[i].label || "") : "";
      const tip = `feat ${i}: ${lbl}\n` +
                  sources.map(s => `  • ${s.kind}: ${s.label}`).join("\n");
      return `<span class="feat-pill ${isMulti ? 'multi' : ''}" title="${esc(tip)}">${i}</span>`;
    };

    let summary = `
      <strong>${ABLATED.size}</strong> features silenced from
      <strong>${groups.length}</strong> source${groups.length === 1 ? '' : 's'}.`;
    if (totalMultiSource > 0) {
      summary += ` <span style="color: var(--hot);"><strong>${totalMultiSource}</strong> feature${totalMultiSource === 1 ? ' was' : 's were'} flagged by 2+ sources (gold pills below).</span>`;
    }
    summary += `<div style="margin-top: 4px; font-size: 12px; color: var(--faint);">Click "show Cypher" to see the actual graph query that recorded this intervention into Neo4j.</div>`;

    for (const g of groups) {
      const idxs = [...g.indices].sort((a, b) => a - b);
      summary += `
        <div class="src-group">
          <div class="src-head">
            <span class="src-kind">${esc(g.kind)}</span>
            <span>${esc(g.label)}</span>
            <span class="src-count">${idxs.length} feature${idxs.length === 1 ? '' : 's'}</span>
          </div>
          <div class="feats">${idxs.map(featurePill).join("")}</div>
        </div>`;
    }
    $("audit-summary").innerHTML = summary;

    // Cypher payload — what would be written to Neo4j to persist this intervention
    const cypherPreview = buildAuditCypher(prompt);
    $("audit-cypher").textContent = cypherPreview;
  }

  function buildAuditCypher(prompt) {
    const interventionId = "i_" + Date.now().toString(36);
    const sources = [...new Set([...ABLATION_SOURCES.values()].flat().map(s => s.kind + "::" + s.label))];
    return `// Persist this intervention into Neo4j
MERGE (s:Session {id: $session_id})
  ON CREATE SET s.started_at = datetime()
CREATE (i:Intervention {
  id:        '${interventionId}',
  prompt:    $prompt,
  ts:        datetime(),
  n_silenced: ${ABLATED.size}
})
CREATE (s)-[:RAN]->(i)
WITH i
UNWIND $sources AS src
  MERGE (so:Source {kind: src.kind, label: src.label, intervention: i.id})
  CREATE (i)-[:USED_SOURCE]->(so)
  WITH i, so, src
  UNWIND src.features AS feat_idx
    MATCH (f:SAEFeature {index: feat_idx})
      WHERE f.sae_id CONTAINS 'L20/16k'
    MERGE (so)-[:SELECTED]->(f)
    MERGE (i)-[:SILENCED]->(f)

// Read it back later:
//   MATCH path = (s:Session {id: '${SESSION_ID}'})
//     -[:RAN]->(:Intervention {id: '${interventionId}'})
//     -[:USED_SOURCE]->(:Source)-[:SELECTED]->(f:SAEFeature)
//   RETURN path`;
  }

  async function persistAudit(prompt, baseline_text, ablated_text) {
    if (ABLATED.size === 0) return;
    // Group features per source for the write
    const bySrc = new Map();
    for (const [idx, sources] of ABLATION_SOURCES.entries()) {
      if (!ABLATED.has(idx)) continue;
      for (const s of sources) {
        const key = s.kind + "::" + s.label;
        if (!bySrc.has(key)) bySrc.set(key, { kind: s.kind, label: s.label, features: [] });
        bySrc.get(key).features.push(idx);
      }
    }
    const payload = {
      cmd: "write_audit",
      session_id: SESSION_ID,
      prompt,
      baseline_text: (baseline_text || "").slice(0, 2000),
      ablated_text:  (ablated_text  || "").slice(0, 2000),
      n_silenced: ABLATED.size,
      sources: [...bySrc.values()],
    };
    try {
      const r = await fetch(PROBE, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).then(r => r.json());
      if (r.ok) LAST_INTERVENTION_ID = r.result.intervention_id;
    } catch (e) { /* swallow — UI still has the client-side audit */ }
  }

  // ─── "See the magic" auto-tour ──────────────────────────────────────────
  // Plays a hand-picked sequence that walks a first-time viewer through the
  // demo's full pitch in ~30 seconds: pick a prompt → search a concept →
  // silence the coalition → generate side-by-side. No prior knowledge.
  async function tour() {
    const $tour = $("tour-btn");
    if ($tour) { $tour.disabled = true; $tour.textContent = "▶ Touring…"; }
    try {
      // 0. Reset
      clearAblation();
      MATCHED.clear();
      $("search").value = "";
      redraw();
      await sleep(400);

      // 1. Set prompt
      const TOUR_PROMPT = "This isn't a setback";
      $("prompt").value = TOUR_PROMPT;
      $("prompt").focus();
      await sleep(900);

      // 2. Highlight "negation" via search — show the graph-backed concept layer
      $("search").value = "negation";
      runSearch("negation");
      await sleep(2200);

      // 3. Swap to the canonical coalition preset (red on the cloud)
      $("search").value = "";
      runSearch("");
      pickPreset("top25");
      $("search-info").textContent = "Now silencing the 25 features that, together, do the construction.";
      await sleep(1700);

      // 4. Generate side-by-side
      await generate();
    } finally {
      if ($tour) { $tour.disabled = false; $tour.textContent = "▶ See the magic"; }
    }
  }

  // ─── wiring ────────────────────────────────────────────────────────────
  function wireControls() {
    $("generate").addEventListener("click", generate);
    $("prompt").addEventListener("keydown", e => { if (e.key === "Enter") generate(); });
    const $tour = $("tour-btn");
    if ($tour) $tour.addEventListener("click", tour);
    const $retrieve = $("retrieve-btn");
    if ($retrieve) $retrieve.addEventListener("click", retrieveConcepts);
    const $surgical = $("surgical-btn");
    if ($surgical) $surgical.addEventListener("click", surgicalDeslop);
    const $cypherToggle = $("audit-cypher-toggle");
    if ($cypherToggle) $cypherToggle.addEventListener("click", () => {
      const c = $("audit-cypher");
      const showing = c.style.display === "block";
      c.style.display = showing ? "none" : "block";
      $cypherToggle.textContent = showing ? "show Cypher" : "hide Cypher";
    });

    // Mixer wiring
    document.querySelectorAll(".pg-slider input[type='range']").forEach(s => {
      const valSpan = s.parentElement.querySelector(".slider-val");
      s.addEventListener("input", () => {
        valSpan.textContent = `${s.value}%`;
        renderMixerInfo();
      });
    });
    const $mixerApply = $("mixer-apply");
    if ($mixerApply) $mixerApply.addEventListener("click", applyMixer);
    const $mixerCypher = $("mixer-cypher-toggle");
    if ($mixerCypher) $mixerCypher.addEventListener("click", () => {
      const c = $("mixer-cypher");
      const showing = c.style.display === "block";
      c.style.display = showing ? "none" : "block";
      $mixerCypher.textContent = showing ? "show Cypher" : "hide Cypher";
    });

    document.querySelectorAll(".preset-card[data-preset]").forEach(card => {
      card.addEventListener("click", () => pickPreset(card.dataset.preset));
    });

    $("clear-ablated").addEventListener("click", clearAblation);
    $("search").addEventListener("input", () => {
      runSearch($("search").value.trim());
    });
    // Submit search on Enter — same as clicking "Silence matches"
    $("search").addEventListener("keydown", e => {
      if (e.key === "Enter" && MATCHED.size > 0) {
        const term = $("search").value.trim();
        addToAblation([...MATCHED], { kind: "search_match",
          label: `search: "${term || '?'}"` });
      }
    });
    $("search-ablate").addEventListener("click", () => {
      if (MATCHED.size === 0) return;
      const term = $("search").value.trim();
      addToAblation([...MATCHED], { kind: "search_match",
        label: `search: "${term || '?'}"` });
    });
    $("search-clear").addEventListener("click", () => {
      $("search").value = "";
      neighbourAnchor = null;
      neighbourPulse = 0;
      runSearch("");
    });
  }

  init().catch(e => {
    console.error(e);
    setStatus("down", "init error: " + e.message);
  });
})();
