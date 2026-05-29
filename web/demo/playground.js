// Playground v3 — friendlier UX, pan/zoom map, community navigator, concept
// chips. Same daemon contract as before. The core intervention is prompt-aware:
// find concepts touched by the prompt, intersect them with a named construction,
// then silence only the overlap. Power users still get click-to-toggle on the map.

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
  let GRAPH_EDGES = [];
  let playSession = 0;
  let activeRun = null;
  let statusTimer = null;
  const GRAPH_CACHE = new Map();
  const ACTIVE_FEATURES = new Set();
  let SUPPORTS_GRAPH_BATCH = false;
  let RELATIONSHIP_RENDER_MODE = "links";
  let MAP_NOTE_TEXT = "";

  // Pan/zoom view state
  let viewScale = 1, viewX = 0, viewY = 0;
  let baseCanvas = null;
  let baseCtx = null;
  let baseKey = "";
  let baseViewX = 0;
  let baseViewY = 0;
  let basePad = 0;
  let graphCommunityCanvas = null;
  let graphCommunityCtx = null;
  let graphCommunityKey = "";
  let graphCommunityPositions = null;
  let graphCommunityMeta = [];
  let redrawQueued = false;
  let lastNearestAt = 0;
  let dragging = false, dragStartX = 0, dragStartY = 0, dragStartViewX = 0, dragStartViewY = 0;
  let dragMoved = false;
  let atlasHitIndex = null;
  let communityHitIndex = null;
  let communityHitIndexKey = "";
  let labelPickCache = new Map();
  let ablatedRenderVersion = 0;

  // Lasso state — shift-drag draws a polygon, on release we add every
  // feature inside it to the ablation set
  let lassoActive = false;
  let lassoPoints = []; // [[x, y] in canvas px], appended every mousemove

  // Highlight a community by id (set when hovering a region in the sidebar)
  let hoveredCid = null;

  // The feature most recently alt-clicked — gets a ring + pulse halo
  let neighbourAnchor = null;
  let neighbourPulse = 0; // 0→1 decaying each frame

  // Relationship lens: co-activation edges drawn on top of the same atlas.
  let REL_CANVAS = null, REL_CTX = null;
  let LOCAL_GRAPH = null;
  let RELATIONSHIP_COMMUNITIES = [];
  let MAP_MODE = "atlas";
  let ATLAS_HUD = "no run yet";
  let localGraphRun = 0;
  let LAST_PROMPT_SLICE = null;
  let LAST_SELECTED_LINKS = null;

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

  const CONSTRUCTION_NAME = "not-this-but-that construction";

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
  function setSilenceInfo(html) {
    const el = $("silence-info");
    if (el) el.innerHTML = html;
  }
  async function probe(payload, { timeoutMs = 45000, signal } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort("timeout"), timeoutMs);
    const abortFromParent = () => controller.abort("cancelled");
    if (signal) {
      if (signal.aborted) controller.abort(signal.reason || "cancelled");
      else signal.addEventListener("abort", abortFromParent, { once: true });
    }
    try {
      const response = await fetch(PROBE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (e) {
      if (controller.signal.aborted) {
        const reason = String(controller.signal.reason || "");
        throw new Error(reason === "timeout" ? "request timed out" : "request cancelled");
      }
      throw e;
    } finally {
      clearTimeout(timer);
      if (signal) signal.removeEventListener("abort", abortFromParent);
    }
  }
  function setBusy(button, busyText) {
    if (!button) return () => {};
    const previous = button.innerHTML;
    button.disabled = true;
    button.innerHTML = busyText;
    return () => { button.disabled = false; button.innerHTML = previous; };
  }
  function describeError(e) {
    return esc(e && e.message ? e.message : "unknown error");
  }
  const DEFAULT_MAP_NOTE = "Feature view keeps decoder-vector positions. Click a dot, group, or preset to add it to Currently silenced.";
  function setMapNote(text = DEFAULT_MAP_NOTE) {
    MAP_NOTE_TEXT = text;
    const el = $("map-note");
    const suffix = relationshipModeSuffix();
    if (el) el.textContent = text + suffix;
  }

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
    setupRelationshipCanvas();
    renderCommunityNav();
    wireControls();
    setMapNote(DEFAULT_MAP_NOTE);
    redraw();
    scheduleGraphCommunityPrebuild();

    // Daemon health
    try {
      const r = await probe({ cmd: "ping" }, { timeoutMs: 8000 });
      if (r.ok) {
        SUPPORTS_GRAPH_BATCH = Array.isArray(r.result.capabilities) &&
          r.result.capabilities.includes("graph.coact_batch");
        setStatus("up", `daemon UP · Gemma 2 2B on ${r.result.device}`);
      }
      else { setStatus("down", "daemon: " + r.error); $("generate").disabled = true; }
    } catch (e) {
      setStatus("down", "daemon down — start with scripts/probe_run.sh start");
      $("generate").disabled = true;
    }

    // Fetch preset sets in the background
    try {
      const r1 = await probe({ cmd: "attribution", top_n: 25, kind: "promote" }, { timeoutMs: 15000 });
      if (r1.ok) TOP25 = r1.result.features;
    } catch (e) {}
    try {
      const r2 = await probe({ cmd: "attribution", top_n: 10, kind: "suppress" }, { timeoutMs: 15000 });
      if (r2.ok) SUPPRESSORS10 = r2.result.features;
    } catch (e) {}
  }

  // ─── canvas + cloud ─────────────────────────────────────────────────────
  function resizeCanvas() {
    if (!CANVAS) return;
    const r = CANVAS.getBoundingClientRect();
    CANVAS.width = r.width * DPR;
    CANVAS.height = r.height * DPR;
    baseKey = "";
    graphCommunityKey = "";
    requestRedraw();
  }

  function setupCanvas() {
    CANVAS = $("canvas");
    CTX = CANVAS.getContext("2d", { alpha: false, desynchronized: true });
    window.addEventListener("resize", resizeCanvas);
    document.addEventListener("fullscreenchange", updateGraphFullscreenState);
    resizeCanvas();
    CANVAS.addEventListener("mousemove", onCanvasMove);
    CANVAS.addEventListener("mousedown", onCanvasDown);
    CANVAS.addEventListener("mouseup", onCanvasUp);
    CANVAS.addEventListener("mouseleave", () => {
      $("tip").style.display = "none";
      dragging = false;
      CANVAS.style.cursor = MAP_MODE === "atlas" ? "crosshair" : "default";
    });
    CANVAS.addEventListener("wheel", onWheel, { passive: false });
    $("zoom-in").addEventListener("click", () => zoomAt(1.4, CANVAS.width/2, CANVAS.height/2));
    $("zoom-out").addEventListener("click", () => zoomAt(1/1.4, CANVAS.width/2, CANVAS.height/2));
    $("zoom-reset").addEventListener("click", () => { viewScale = 1; viewX = 0; viewY = 0; baseKey = ""; requestRedraw(); });
    const fullscreen = $("graph-fullscreen");
    if (fullscreen) fullscreen.addEventListener("click", toggleGraphFullscreen);
  }

  async function toggleGraphFullscreen() {
    const pane = document.querySelector(".pg-right");
    if (!pane) return;
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await pane.requestFullscreen();
    } catch (e) {
      console.warn("fullscreen failed", e);
    }
  }

  function updateGraphFullscreenState() {
    const pane = document.querySelector(".pg-right");
    const on = !!(pane && document.fullscreenElement === pane);
    if (pane) pane.classList.toggle("is-fullscreen", on);
    const btn = $("graph-fullscreen");
    if (btn) {
      btn.textContent = on ? "↙" : "⛶";
      btn.title = on ? "Exit fullscreen" : "Fullscreen graph pane";
    }
    setTimeout(resizeCanvas, 60);
  }

  function setMapMode(mode) {
    MAP_MODE = "atlas";
    const wrap = document.querySelector(".pg-canvas-wrap");
    if (wrap) wrap.classList.remove("relationship-mode");
    $("tip").style.display = "none";
    CANVAS.style.cursor = "crosshair";
    requestRedraw();
  }

  // World coords are layout's [-1,1] space. xy() maps to canvas pixels using
  // the current pan/zoom view state.
  function xy(f) {
    const W = CANVAS.width, H = CANVAS.height;
    const m = Math.min(W, H) * 0.46 * viewScale;
    return [W / 2 + viewX + f.x * m, H / 2 + viewY + f.y * m];
  }
  function screenPointForFeature(idx) {
    if (RELATIONSHIP_RENDER_MODE === "communities") {
      const layout = ensureGraphCommunityLayout(CANVAS.width, CANVAS.height);
      if (!layout || !layout.positions || !layout.positions[idx]) return null;
      return transformCommunityPoint(layout.positions[idx], communityViewTransform(CANVAS.width, CANVAS.height));
    }
    const f = LAYOUT.features[idx];
    return f ? xy(f) : null;
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
    baseKey = "";
    requestRedraw();
  }
  function onWheel(ev) {
    ev.preventDefault();
    const r = CANVAS.getBoundingClientRect();
    const cx = (ev.clientX - r.left) * DPR;
    const cy = (ev.clientY - r.top) * DPR;
    const factor = ev.deltaY < 0 ? 1.12 : 1/1.12;
    zoomAt(factor, cx, cy);
  }

  const COMMUNITY_TONES = [
    { h: 206, s: 38, l: 50 },
    { h: 24, s: 42, l: 52 },
    { h: 326, s: 36, l: 57 },
    { h: 45, s: 38, l: 52 },
    { h: 188, s: 34, l: 48 },
    { h: 265, s: 32, l: 56 },
    { h: 4, s: 36, l: 58 },
    { h: 96, s: 26, l: 48 },
    { h: 226, s: 34, l: 58 },
    { h: 166, s: 30, l: 48 },
    { h: 300, s: 30, l: 55 },
    { h: 33, s: 34, l: 49 },
  ];
  const COMMUNITY_HUE = {};
  function communityTone(cid) {
    const n = Number.isFinite(Number(cid)) ? Math.abs(Number(cid)) : 0;
    if (cid === 12) return { h: 36, s: 42, l: 50 };
    return COMMUNITY_TONES[n % COMMUNITY_TONES.length];
  }
  function hueForCid(cid) {
    return communityTone(cid).h;
  }
  function commColor(cid, alpha = 0.5) {
    if (!(cid in COMMUNITY_HUE)) COMMUNITY_HUE[cid] = communityTone(cid);
    const tone = COMMUNITY_HUE[cid];
    return `hsla(${tone.h}, ${tone.s}%, ${tone.l}%, ${alpha})`;
  }

  function redraw() {
    if (!LAYOUT || !CANVAS) return;
    const W = CANVAS.width, H = CANVAS.height;
    CTX.clearRect(0, 0, W, H);
    const feats = LAYOUT.features;
    const rBase = (0.85 + Math.sqrt(viewScale) * 0.3) * DPR;
    decayActivity();
    if (RELATIONSHIP_RENDER_MODE === "communities") {
      drawRelationshipCommunityLayout(W, H);
      if (lassoActive && lassoPoints.length >= 2) drawLasso();
      return;
    }
    drawBaseAtlas(W, H, feats, rBase);
    drawActivityOverlay(idx => xy(feats[idx]), rBase);
    drawRegionFocus(W, H);
    for (const i of ABLATED) {
      const [x, y] = xy(feats[i]);
      if (x < -10 || x > W + 10 || y < -10 || y > H + 10) continue;
      CTX.fillStyle = "rgba(183, 60, 42, 0.95)";
      CTX.beginPath();
      CTX.arc(x, y, rBase * 2.7, 0, Math.PI * 2);
      CTX.fill();
      CTX.strokeStyle = "rgba(183, 60, 42, 0.42)";
      CTX.lineWidth = 1 * DPR;
      CTX.beginPath();
      CTX.arc(x, y, 5 * DPR, 0, Math.PI * 2);
      CTX.stroke();
    }
    // Bright ring around every MATCHED (search-result / alt-click linked) feature
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
    drawGraphEdges();
    if (neighbourAnchor !== null && LAYOUT.features[neighbourAnchor]) {
      const [x, y] = xy(LAYOUT.features[neighbourAnchor]);
      const r = (8 + 6 * neighbourPulse) * DPR;
      CTX.strokeStyle = `rgba(74, 111, 165, ${0.4 + 0.5 * neighbourPulse})`;
      CTX.lineWidth = 2 * DPR;
      CTX.beginPath();
      CTX.arc(x, y, r, 0, Math.PI * 2);
      CTX.stroke();
      CTX.fillStyle = "rgba(74, 111, 165, 0.95)";
      CTX.beginPath();
      CTX.arc(x, y, 3.5 * DPR, 0, Math.PI * 2);
      CTX.fill();
      neighbourPulse = Math.max(0, neighbourPulse - 0.04);
      if (neighbourPulse > 0) startAnim();
    }
    drawRelationshipLabels();
    if (lassoActive && lassoPoints.length >= 2) drawLasso();
  }

  function decayActivity() {
    if (!ACTIVITY) return;
    for (const i of [...ACTIVE_FEATURES]) {
      ACTIVITY[i] *= 0.93;
      if (ACTIVITY[i] < 0.025) ACTIVE_FEATURES.delete(i);
    }
  }

  function drawActivityOverlay(project, rBase) {
    if (!ACTIVITY || ACTIVE_FEATURES.size === 0) return;
    const W = CANVAS.width, H = CANVAS.height;
    CTX.save();
    for (const i of ACTIVE_FEATURES) {
      if (ABLATED.has(i)) continue;
      const act = ACTIVITY[i];
      if (act <= 0.05) continue;
      const p = project(i);
      if (!p) continue;
      const [x, y] = p;
      if (x < -28 || x > W + 28 || y < -28 || y > H + 28) continue;
      if (act > 0.68) {
        const halo = Math.max(18 * DPR, rBase * (8.0 + 5.0 * act));
        CTX.fillStyle = `rgba(214, 133, 36, ${0.12 + 0.18 * act})`;
        CTX.beginPath();
        CTX.arc(x, y, halo, 0, Math.PI * 2);
        CTX.fill();
        CTX.strokeStyle = "rgba(255,255,255,0.96)";
        CTX.lineWidth = 2.8 * DPR;
        CTX.beginPath();
        CTX.arc(x, y, Math.max(8.0 * DPR, rBase * 5.8), 0, Math.PI * 2);
        CTX.stroke();
        CTX.fillStyle = "rgba(214, 133, 36, 0.98)";
        CTX.beginPath();
        CTX.arc(x, y, Math.max(5.8 * DPR, rBase * 3.8), 0, Math.PI * 2);
        CTX.fill();
        CTX.strokeStyle = "rgba(112, 62, 34, 0.68)";
        CTX.lineWidth = 1.25 * DPR;
        CTX.stroke();
      } else {
        const a = 0.34 + 0.45 * act;
        CTX.fillStyle = `rgba(88, 127, 137, ${a.toFixed(2)})`;
        CTX.beginPath();
        CTX.arc(x, y, Math.max(2.8 * DPR, rBase * (1.8 + 1.5 * act)), 0, Math.PI * 2);
        CTX.fill();
      }
    }
    CTX.restore();
  }

  function drawBaseAtlas(W, H, feats, rBase) {
    const pad = Math.round(Math.max(96 * DPR, Math.min(320 * DPR, Math.min(W, H) * 0.28)));
    const CW = W + pad * 2;
    const CH = H + pad * 2;
    const scaleKey = viewScale.toFixed(4);
    const hoverKey = hoveredCid ?? "none";
    const key = [W, H, scaleKey, hoverKey].join(":");
    const sizeMatches = !!baseCanvas && baseCanvas.width === CW && baseCanvas.height === CH;
    const canReusePan =
      sizeMatches &&
      baseKey === key &&
      Math.abs(viewX - baseViewX) <= pad * 0.72 &&
      Math.abs(viewY - baseViewY) <= pad * 0.72;
    if (!sizeMatches) {
      baseCanvas = document.createElement("canvas");
      baseCanvas.width = CW;
      baseCanvas.height = CH;
      baseCtx = baseCanvas.getContext("2d", { alpha: false, desynchronized: true });
      baseKey = "";
    }
    if (!canReusePan) {
      baseViewX = viewX;
      baseViewY = viewY;
      basePad = pad;
      baseCtx.clearRect(0, 0, CW, CH);
      baseCtx.fillStyle = "#fff";
      baseCtx.fillRect(0, 0, CW, CH);
      const m = Math.min(W, H) * 0.46 * viewScale;
      for (let i = 0; i < feats.length; i++) {
        const f = feats[i];
        const x = pad + W / 2 + baseViewX + f.x * m;
        const y = pad + H / 2 + baseViewY + f.y * m;
        if (x < -10 || x > CW + 10 || y < -10 || y > CH + 10) continue;
        const isHoveredCommunity = hoveredCid !== null && f.cid === hoveredCid;
        const alpha = Math.min(0.62, 0.30 + Math.log1p(viewScale) * 0.16);
        baseCtx.fillStyle = isHoveredCommunity ? commColor(f.cid, 0.86) : commColor(f.cid, alpha);
        const r = isHoveredCommunity ? rBase * 2.25 : Math.min(3.0 * DPR, rBase * 1.24);
        baseCtx.beginPath();
        baseCtx.arc(x, y, r, 0, Math.PI * 2);
        baseCtx.fill();
      }
      baseKey = key;
    }
    CTX.drawImage(baseCanvas, viewX - baseViewX - basePad, viewY - baseViewY - basePad);
  }

  function drawLasso() {
    CTX.strokeStyle = "rgba(168, 93, 57, 0.85)";
    CTX.lineWidth = 1.4 * DPR;
    CTX.fillStyle = "rgba(168, 93, 57, 0.10)";
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
      CTX.fillStyle = "rgba(168, 93, 57, 0.95)";
      CTX.textAlign = "left";
      CTX.textBaseline = "bottom";
      CTX.fillText(`${inside} features`, cx + 8*DPR, cy - 4*DPR);
    }
  }

  // ─── Region focus rendering ───────────────────────────────────────────
  // Communities can be diffuse and non-convex in the UMAP. Avoid drawing
  // fake region boundaries; on hover, emphasize the actual member dots and
  // place a small centroid label instead.
  let COMM_STATS = null;
  function computeCommunityStats() {
    if (COMM_STATS) return COMM_STATS;
    const byCid = {};
    for (const f of LAYOUT.features) {
      const slot = byCid[f.cid] || (byCid[f.cid] = { xs: [], ys: [], cid: f.cid, members: [] });
      slot.xs.push(f.x); slot.ys.push(f.y); slot.members.push(f.idx);
    }
    const out = {};
    for (const cid in byCid) {
      const s = byCid[cid];
      const n = s.xs.length;
      const cx = s.xs.reduce((a, b) => a + b, 0) / n;
      const cy = s.ys.reduce((a, b) => a + b, 0) / n;
      out[cid] = {
        cid: +cid, n, cx, cy,
        members: s.members,
        name: (COMM && COMM[cid]) ? COMM[cid].name : `community ${cid}`,
      };
    }
    COMM_STATS = out;
    return out;
  }

  function drawRegionFocus(W, H) {
    if (hoveredCid === null || !COMM) return;
    const stats = computeCommunityStats();
    const s = stats[hoveredCid];
    if (!s) return;
    const m = Math.min(W, H) * 0.46 * viewScale;
    const cx = W / 2 + viewX + s.cx * m;
    const cy = H / 2 + viewY + s.cy * m;
    if (cx < -160 * DPR || cx > W + 160 * DPR || cy < -80 * DPR || cy > H + 80 * DPR) return;

    const label = s.name.length > 36 ? `${s.name.slice(0, 34)}…` : s.name;
    const text = `${label} · ${s.n}`;
    CTX.save();
    CTX.font = `600 ${12 * DPR}px -apple-system, BlinkMacSystemFont, "Inter", sans-serif`;
    const padX = 9 * DPR;
    const padY = 6 * DPR;
    const textW = CTX.measureText(text).width;
    const boxW = textW + padX * 2;
    const boxH = 26 * DPR;
    const x = Math.max(10 * DPR, Math.min(cx + 12 * DPR, W - boxW - 10 * DPR));
    const y = Math.max(10 * DPR, Math.min(cy - boxH - 12 * DPR, H - boxH - 10 * DPR));
    CTX.fillStyle = "rgba(255, 255, 255, 0.94)";
    CTX.strokeStyle = commColor(s.cid, 0.65);
    CTX.lineWidth = 1 * DPR;
    roundRect(x, y, boxW, boxH, 5 * DPR);
    CTX.fill();
    CTX.stroke();
    CTX.fillStyle = commColor(s.cid, 0.95);
    CTX.textBaseline = "middle";
    CTX.textAlign = "left";
    CTX.fillText(text, x + padX, y + boxH / 2);
    CTX.restore();
  }

  function roundRect(x, y, w, h, r) {
    CTX.beginPath();
    CTX.moveTo(x + r, y);
    CTX.arcTo(x + w, y, x + w, y + h, r);
    CTX.arcTo(x + w, y + h, x, y + h, r);
    CTX.arcTo(x, y + h, x, y, r);
    CTX.arcTo(x, y, x + w, y, r);
    CTX.closePath();
  }

  let animHandle = null;
  function requestRedraw() {
    if (animHandle || redrawQueued) return;
    redrawQueued = true;
    requestAnimationFrame(() => {
      redrawQueued = false;
      redraw();
    });
  }
  function startAnim() {
    if (animHandle) return;
    redrawQueued = false;
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
    const threshold = 14 * DPR;
    return RELATIONSHIP_RENDER_MODE === "communities"
      ? nearestCommunityFeature(mx, my, threshold)
      : nearestAtlasFeature(mx, my, threshold);
  }

  function ensureAtlasHitIndex() {
    const feats = LAYOUT && LAYOUT.features;
    if (!feats) return null;
    if (atlasHitIndex && atlasHitIndex.n === feats.length) return atlasHitIndex;
    const cell = 0.035;
    const bins = new Map();
    for (let i = 0; i < feats.length; i++) {
      const f = feats[i];
      const cx = Math.floor(f.x / cell);
      const cy = Math.floor(f.y / cell);
      const key = `${cx},${cy}`;
      let bucket = bins.get(key);
      if (!bucket) bins.set(key, bucket = []);
      bucket.push(i);
    }
    atlasHitIndex = { n: feats.length, cell, bins };
    return atlasHitIndex;
  }

  function nearestAtlasFeature(mx, my, threshold) {
    const index = ensureAtlasHitIndex();
    if (!index) return null;
    const W = CANVAS.width, H = CANVAS.height;
    const m = Math.min(W, H) * 0.46 * viewScale;
    const wx = (mx - W / 2 - viewX) / m;
    const wy = (my - H / 2 - viewY) / m;
    const radius = Math.max(1, Math.ceil((threshold / m) / index.cell));
    const cx = Math.floor(wx / index.cell);
    const cy = Math.floor(wy / index.cell);
    let nearest = null, nd = 1e9;
    const feats = LAYOUT.features;
    for (let gx = cx - radius; gx <= cx + radius; gx++) {
      for (let gy = cy - radius; gy <= cy + radius; gy++) {
        const bucket = index.bins.get(`${gx},${gy}`);
        if (!bucket) continue;
        for (const i of bucket) {
          const f = feats[i];
          const x = W / 2 + viewX + f.x * m;
          const y = H / 2 + viewY + f.y * m;
          const d = (x - mx) ** 2 + (y - my) ** 2;
          if (d < nd) { nd = d; nearest = i; }
        }
      }
    }
    return nd < threshold * threshold ? nearest : null;
  }

  function ensureCommunityHitIndex(layout) {
    const positions = layout && layout.positions;
    if (!positions) return null;
    const key = `${graphCommunityKey}:${positions.length}:${CANVAS.width}:${CANVAS.height}`;
    if (communityHitIndex && communityHitIndexKey === key) return communityHitIndex;
    const cell = 24 * DPR;
    const bins = new Map();
    for (let i = 0; i < positions.length; i++) {
      const p = positions[i];
      if (!p) continue;
      const cx = Math.floor(p[0] / cell);
      const cy = Math.floor(p[1] / cell);
      const binKey = `${cx},${cy}`;
      let bucket = bins.get(binKey);
      if (!bucket) bins.set(binKey, bucket = []);
      bucket.push(i);
    }
    communityHitIndexKey = key;
    communityHitIndex = { cell, bins };
    return communityHitIndex;
  }

  function nearestCommunityFeature(mx, my, threshold) {
    const layout = ensureGraphCommunityLayout(CANVAS.width, CANVAS.height);
    const index = ensureCommunityHitIndex(layout);
    if (!layout || !index) return null;
    const t = communityViewTransform(CANVAS.width, CANVAS.height);
    const rawX = (mx - t.x) / t.scale;
    const rawY = (my - t.y) / t.scale;
    const rawThreshold = threshold / t.scale;
    const radius = Math.max(1, Math.ceil(rawThreshold / index.cell));
    const cx = Math.floor(rawX / index.cell);
    const cy = Math.floor(rawY / index.cell);
    let nearest = null, nd = 1e9;
    for (let gx = cx - radius; gx <= cx + radius; gx++) {
      for (let gy = cy - radius; gy <= cy + radius; gy++) {
        const bucket = index.bins.get(`${gx},${gy}`);
        if (!bucket) continue;
        for (const i of bucket) {
          const p = layout.positions[i];
          if (!p) continue;
          const d = (p[0] - rawX) ** 2 + (p[1] - rawY) ** 2;
          if (d < nd) { nd = d; nearest = i; }
        }
      }
    }
    return nd < rawThreshold * rawThreshold ? nearest : null;
  }

  function onCanvasDown(ev) {
    if (ev.shiftKey) {
      // Shift-drag: lasso selection
      lassoActive = true;
      const rect = CANVAS.getBoundingClientRect();
      lassoPoints = [[(ev.clientX - rect.left) * DPR, (ev.clientY - rect.top) * DPR]];
      CANVAS.style.cursor = "crosshair";
      $("tip").style.display = "none";
      requestRedraw();
      return;
    }
    dragging = true; dragMoved = false;
    dragStartX = ev.clientX; dragStartY = ev.clientY;
    dragStartViewX = viewX; dragStartViewY = viewY;
    $("tip").style.display = "none";
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
      requestRedraw();
      return;
    }
    if (dragging && !dragMoved) {
      const i = nearestFeature(ev);
      if (i !== null) {
        if (ev.altKey || ev.metaKey) {
          showLinkedFeatures(i);
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
      requestRedraw();
      return;
    }
    if (dragging) {
      const dx = (ev.clientX - dragStartX) * DPR;
      const dy = (ev.clientY - dragStartY) * DPR;
      if (dx*dx + dy*dy > 9) dragMoved = true;
      viewX = dragStartViewX + dx;
      viewY = dragStartViewY + dy;
      requestRedraw();
      return;
    }
    const now = performance.now();
    if (now - lastNearestAt < 32) return;
    lastNearestAt = now;
    const i = nearestFeature(ev);
    const tip = $("tip");
    if (i === null) { tip.style.display = "none"; return; }
    const f = LAYOUT.features[i];
    const commName = COMM && COMM[f.cid] ? COMM[f.cid].name : `community ${f.cid}`;
    const rect = CANVAS.getBoundingClientRect();
    tip.style.display = "block";
    tip.innerHTML = `<span class="idx">feature ${f.idx}</span>
      <span class="lbl">${esc(f.label) || "(no auto-interp label)"}</span>
      <div class="meta">in ${esc(commName)} · density ${(f.density*100).toFixed(1)}%</div>
      <div class="tip-actions">
        <span><kbd>click</kbd> ${ABLATED.has(f.idx) ? "unsilence" : "silence"}</span>
        <span><kbd>⌥ click</kbd> relationships</span>
        <span><kbd>⇧ drag</kbd> lasso</span>
      </div>`;
    positionTip(tip, ev, rect);
  }

  function positionTip(tip, ev, rect) {
    const margin = 12;
    tip.style.left = "0px";
    tip.style.top = "0px";
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = ev.clientX - rect.left + 14;
    let top = ev.clientY - rect.top + 12;
    if (left + w + margin > rect.width) left = ev.clientX - rect.left - w - 14;
    if (top + h + margin > rect.height) top = ev.clientY - rect.top - h - 14;
    left = Math.max(margin, Math.min(left, rect.width - w - margin));
    top = Math.max(margin, Math.min(top, rect.height - h - margin));
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  function drawGraphEdges(project = null) {
    if (!GRAPH_EDGES.length) return;
    const feats = LAYOUT.features;
    const point = project || (idx => {
      const f = feats[idx];
      return f ? xy(f) : null;
    });
    CTX.save();
    CTX.lineCap = "round";
    for (const edge of GRAPH_EDGES) {
      const from = feats[edge.from];
      const target = feats[edge.to];
      if (!from || !target) continue;
      const a = point(edge.from);
      const b = point(edge.to);
      if (!a || !b) continue;
      const [fx, fy] = a;
      const [tx, ty] = b;
      if ((fx < -80 && tx < -80) || (fx > CANVAS.width + 80 && tx > CANVAS.width + 80) ||
          (fy < -80 && ty < -80) || (fy > CANVAS.height + 80 && ty > CANVAS.height + 80)) continue;
      const score = Number.isFinite(edge.score) ? Math.max(0, Math.min(1, edge.score)) : 0.45;
      CTX.strokeStyle = "rgba(255, 255, 255, 0.72)";
      CTX.lineWidth = (2.4 + score * 1.8) * DPR;
      CTX.beginPath();
      CTX.moveTo(fx, fy);
      CTX.lineTo(tx, ty);
      CTX.stroke();
      CTX.strokeStyle = `rgba(38, 91, 166, ${0.38 + score * 0.26})`;
      CTX.lineWidth = (0.9 + score * 1.4) * DPR;
      CTX.beginPath();
      CTX.moveTo(fx, fy);
      CTX.lineTo(tx, ty);
      CTX.stroke();
    }
    CTX.restore();
  }

  function drawRelationshipCommunityLayout(W, H) {
    const layout = ensureGraphCommunityLayout(W, H);
    if (!layout) return;
    const { positions: rawPositions, groups: rawGroups } = layout;
    const t = communityViewTransform(W, H);
    const groups = transformCommunityGroups(rawGroups, t);
    const needsSharpRedraw = t.scale > 1.08 || t.scale < 0.96;
    const positions = needsSharpRedraw
      ? transformCommunityPositions(rawPositions, t)
      : idx => {
        const p = rawPositions[idx];
        return p ? transformCommunityPoint(p, t) : null;
      };
    CTX.fillStyle = "#fff";
    CTX.fillRect(0, 0, W, H);
    if (needsSharpRedraw) drawCommunityBaseVector(positions, groups, W, H);
    else CTX.drawImage(graphCommunityCanvas, t.x, t.y, W * t.scale, H * t.scale);
    drawCommunityLabels(groups);
    drawCommunityHoverOverlay(positions, groups);
    drawGraphEdges(idx => communityPosition(positions, idx));
    drawActivityOverlay(idx => communityPosition(positions, idx), (0.85 + Math.sqrt(viewScale) * 0.3) * DPR);
    drawCommunityHighlights(positions);
    drawCommunityFeatureLabels(positions);
  }

  function communityPosition(positions, idx) {
    return typeof positions === "function" ? positions(idx) : (positions[idx] || null);
  }

  function drawCommunityBaseVector(positions, groups, W, H) {
    CTX.save();
    for (const g of groups) {
      if (g.x + g.r < -80 || g.x - g.r > W + 80 || g.y + g.r < -80 || g.y - g.r > H + 80) continue;
      const glowR = Math.min(g.r * 0.62, 260 * DPR);
      const grad = CTX.createRadialGradient(g.x, g.y, 0, g.x, g.y, glowR);
      grad.addColorStop(0, commColor(g.cid, 0.045));
      grad.addColorStop(1, commColor(g.cid, 0));
      CTX.fillStyle = grad;
      CTX.beginPath();
      CTX.arc(g.x, g.y, glowR, 0, Math.PI * 2);
      CTX.fill();
    }

    const dotR = Math.min(2.8 * DPR, Math.max(1.15 * DPR, (0.95 + Math.log2(Math.max(1, viewScale)) * 0.38) * DPR));
    for (const g of groups) {
      CTX.fillStyle = commColor(g.cid, hoveredCid === g.cid ? 0.72 : 0.48);
      for (const idx of g.members) {
        const p = positions[idx];
        if (!p) continue;
        const [x, y] = p;
        if (x < -8 || x > W + 8 || y < -8 || y > H + 8) continue;
        CTX.beginPath();
        CTX.arc(x, y, dotR, 0, Math.PI * 2);
        CTX.fill();
      }
    }
    CTX.restore();
  }

  function communityViewTransform(W, H) {
    return {
      scale: viewScale,
      x: W / 2 + viewX - (W / 2) * viewScale,
      y: H / 2 + viewY - (H / 2) * viewScale,
    };
  }

  function transformCommunityPoint(p, t) {
    return [t.x + p[0] * t.scale, t.y + p[1] * t.scale];
  }

  function transformCommunityPositions(rawPositions, t) {
    const out = new Array(rawPositions.length);
    for (let i = 0; i < rawPositions.length; i++) {
      const p = rawPositions[i];
      if (p) out[i] = transformCommunityPoint(p, t);
    }
    return out;
  }

  function transformCommunityGroups(rawGroups, t) {
    return rawGroups.map(g => ({
      ...g,
      x: t.x + g.x * t.scale,
      y: t.y + g.y * t.scale,
      r: g.r * t.scale,
    }));
  }

  function ensureGraphCommunityLayout(W, H) {
    if (!LAYOUT || !LAYOUT.features) return null;
    const key = `organic-v3:${W}:${H}:${LAYOUT.features.length}`;
    if (graphCommunityKey === key && graphCommunityPositions) {
      return { positions: graphCommunityPositions, groups: graphCommunityMeta };
    }
    const feats = LAYOUT.features;
    const byCid = new Map();
    for (const f of feats) {
      const cid = Number.isFinite(Number(f.cid)) ? Number(f.cid) : -1;
      if (!byCid.has(cid)) byCid.set(cid, []);
      byCid.get(cid).push(f.idx);
    }
    const groups = [...byCid.entries()]
      .map(([cid, members]) => ({ cid, members }))
      .sort((a, b) => b.members.length - a.members.length);
    const maxN = Math.max(1, groups[0] ? groups[0].members.length : 1);
    const minDim = Math.min(W, H);
    const jitter = (idx, salt) => {
      const x = Math.sin((idx + 1) * 12.9898 + salt * 78.233) * 43758.5453;
      return x - Math.floor(x);
    };
    for (let i = 0; i < groups.length; i++) {
      const g = groups[i];
      g.colorIndex = i;
      let sx = 0, sy = 0;
      for (const idx of g.members) {
        const f = feats[idx];
        sx += f.x;
        sy += f.y;
      }
      g.cx0 = sx / Math.max(1, g.members.length);
      g.cy0 = sy / Math.max(1, g.members.length);
      const distances = g.members.map(idx => {
        const f = feats[idx];
        return Math.hypot(f.x - g.cx0, f.y - g.cy0);
      }).sort((a, b) => a - b);
      g.spread = Math.max(0.035, distances[Math.floor(distances.length * 0.88)] || 0.035);
      g.r = (34 + Math.sqrt(g.members.length / maxN) * 118) * DPR;
      g.x = W / 2 + g.cx0 * minDim * 0.34;
      g.y = H / 2 + g.cy0 * minDim * 0.34;
    }
    const margin = 24 * DPR;
    for (let iter = 0; iter < 120; iter++) {
      for (let i = 0; i < groups.length; i++) {
        for (let j = i + 1; j < groups.length; j++) {
          const a = groups[i], b = groups[j];
          let dx = b.x - a.x, dy = b.y - a.y;
          let d = Math.sqrt(dx * dx + dy * dy) || 1;
          const target = a.r + b.r + 18 * DPR;
          if (d >= target) continue;
          const push = (target - d) * 0.45;
          dx /= d; dy /= d;
          a.x -= dx * push; a.y -= dy * push;
          b.x += dx * push; b.y += dy * push;
        }
      }
      for (const g of groups) {
        const targetX = W / 2 + g.cx0 * minDim * 0.30;
        const targetY = H / 2 + g.cy0 * minDim * 0.30;
        g.x += (targetX - g.x) * 0.012;
        g.y += (targetY - g.y) * 0.012;
        g.x = Math.max(margin + g.r, Math.min(W - margin - g.r, g.x));
        g.y = Math.max(margin + g.r, Math.min(H - margin - g.r, g.y));
      }
    }

    const positions = new Array(feats.length);
    for (let gi = 0; gi < groups.length; gi++) {
      const g = groups[gi];
      const scale = (g.r * 0.82) / g.spread;
      for (const idx of g.members) {
        const f = feats[idx];
        let dx = f.x - g.cx0;
        let dy = f.y - g.cy0;
        const dist = Math.hypot(dx, dy);
        if (dist > 0) {
          const eased = g.spread * 1.42 * Math.tanh(dist / (g.spread * 1.42));
          const k = eased / dist;
          dx *= k;
          dy *= k;
        }
        const localJitter = Math.max(0.35, 1.35 - Math.sqrt(g.members.length / maxN)) * DPR;
        positions[idx] = [
          g.x + dx * scale + (jitter(idx, 1) - 0.5) * localJitter * 5.5,
          g.y + dy * scale + (jitter(idx, 2) - 0.5) * localJitter * 5.5,
        ];
      }
    }

    graphCommunityCanvas = document.createElement("canvas");
    graphCommunityCanvas.width = W;
    graphCommunityCanvas.height = H;
    graphCommunityCtx = graphCommunityCanvas.getContext("2d", { alpha: false, desynchronized: true });
    graphCommunityCtx.fillStyle = "#fff";
    graphCommunityCtx.fillRect(0, 0, W, H);
    graphCommunityCtx.save();
    for (let gi = 0; gi < groups.length; gi++) {
      const g = groups[gi];
      for (let k = 0; k < 3; k++) {
        const a = (gi * 1.9 + k * 2.1) % (Math.PI * 2);
        const cx = g.x + Math.cos(a) * g.r * (0.13 + k * 0.04);
        const cy = g.y + Math.sin(a) * g.r * (0.09 + k * 0.03);
        const grad = graphCommunityCtx.createRadialGradient(cx, cy, 0, cx, cy, g.r * (0.70 + k * 0.10));
        grad.addColorStop(0, commColor(g.cid, 0.055));
        grad.addColorStop(1, commColor(g.cid, 0));
        graphCommunityCtx.fillStyle = grad;
        graphCommunityCtx.beginPath();
        graphCommunityCtx.arc(cx, cy, g.r * (0.72 + k * 0.10), 0, Math.PI * 2);
        graphCommunityCtx.fill();
      }
    }
    for (let gi = 0; gi < groups.length; gi++) {
      const g = groups[gi];
      graphCommunityCtx.fillStyle = commColor(g.cid, 0.42);
      for (const idx of g.members) {
        const p = positions[idx];
        if (!p) continue;
        graphCommunityCtx.beginPath();
        graphCommunityCtx.arc(p[0], p[1], 1.2 * DPR, 0, Math.PI * 2);
        graphCommunityCtx.fill();
      }
    }
    graphCommunityCtx.restore();

    graphCommunityKey = key;
    graphCommunityPositions = positions;
    graphCommunityMeta = groups;
    return { positions, groups };
  }

  function drawCommunityLabels(groups) {
    const W = CANVAS.width, H = CANVAS.height;
    const visible = groups
      .filter(g => g.x + g.r > 0 && g.x - g.r < W && g.y + g.r > 0 && g.y - g.r < H)
      .sort((a, b) => b.members.length - a.members.length);
    const placed = [];
    const picks = visible.filter(g => hoveredCid === g.cid).concat(visible.slice(0, 7));
    CTX.save();
    for (const g of picks) {
      if (!g || g._labelDone) continue;
      g._labelDone = true;
      const box = measureCommunityLabel(g);
      const bx = Math.max(8 * DPR, Math.min(g.x - box.w / 2, W - box.w - 8 * DPR));
      const by = Math.max(60 * DPR, Math.min(g.y - Math.min(g.r, 150 * DPR) * 0.72 - box.h - 4 * DPR, H - box.h - 8 * DPR));
      const rect = { x: bx, y: by, w: box.w, h: box.h };
      const overlaps = placed.some(r => !(rect.x + rect.w < r.x || r.x + r.w < rect.x || rect.y + rect.h < r.y || r.y + r.h < rect.y));
      if (overlaps && hoveredCid !== g.cid) continue;
      drawCommunityLabelAt(CTX, g, bx, by, box, hoveredCid === g.cid ? 0.96 : 0.76);
      placed.push(rect);
    }
    for (const g of groups) delete g._labelDone;
    CTX.restore();
  }

  function measureCommunityLabel(g) {
    const name = COMM && COMM[g.cid] ? COMM[g.cid].name : `community ${g.cid}`;
    const label = `${name.slice(0, 30)} · ${g.members.length}`;
    const labelPad = 10 * DPR;
    const boxH = 23 * DPR;
    CTX.font = `700 ${10.5 * DPR}px -apple-system, BlinkMacSystemFont, "Inter", sans-serif`;
    return { label, pad: labelPad, h: boxH, w: CTX.measureText(label).width + labelPad * 2 };
  }

  function drawCommunityLabelOn(ctx, g, W, H, alpha = 0.82) {
    const box = measureCommunityLabel(g);
    const bx = Math.max(8 * DPR, Math.min(g.x - box.w / 2, W - box.w - 8 * DPR));
    const by = Math.max(60 * DPR, Math.min(g.y - Math.min(g.r, 150 * DPR) * 0.72 - box.h - 4 * DPR, H - box.h - 8 * DPR));
    drawCommunityLabelAt(ctx, g, bx, by, box, alpha);
  }

  function drawCommunityLabelAt(ctx, g, bx, by, box, alpha = 0.82) {
    const name = COMM && COMM[g.cid] ? COMM[g.cid].name : `community ${g.cid}`;
    const label = `${name.slice(0, 30)} · ${g.members.length}`;
    ctx.font = `700 ${10.5 * DPR}px -apple-system, BlinkMacSystemFont, "Inter", sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.strokeStyle = commColor(g.cid, Math.min(0.86, alpha));
    ctx.lineWidth = 1 * DPR;
    roundRectOn(ctx, bx, by, box.w, box.h, 4 * DPR);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = `rgba(36,35,32,${Math.min(0.9, alpha + 0.1)})`;
    ctx.fillText(label, bx + box.pad, by + box.h / 2);
  }

  function drawCommunityHoverOverlay(positions, groups) {
    if (hoveredCid === null) return;
    const g = groups.find(group => group.cid === hoveredCid);
    if (!g) return;
    CTX.save();
    CTX.fillStyle = commColor(g.cid, 0.12);
    for (const idx of g.members) {
      const p = communityPosition(positions, idx);
      if (!p) continue;
      CTX.beginPath();
      CTX.arc(p[0], p[1], 5.2 * DPR, 0, Math.PI * 2);
      CTX.fill();
    }
    CTX.fillStyle = commColor(g.cid, 0.92);
    for (const idx of g.members) {
      const p = communityPosition(positions, idx);
      if (!p) continue;
      CTX.beginPath();
      CTX.arc(p[0], p[1], 2.3 * DPR, 0, Math.PI * 2);
      CTX.fill();
    }
    drawCommunityLabelOn(CTX, g, CANVAS.width, CANVAS.height, 0.96);
    CTX.restore();
  }

  function drawCommunityHighlights(positions) {
    const drawNode = (idx, fill, stroke, r) => {
      const p = communityPosition(positions, idx);
      if (!p) return;
      CTX.fillStyle = fill;
      CTX.strokeStyle = stroke;
      CTX.lineWidth = 1.5 * DPR;
      CTX.beginPath();
      CTX.arc(p[0], p[1], r * DPR, 0, Math.PI * 2);
      CTX.fill();
      CTX.stroke();
    };
    for (const idx of MATCHED) {
      if (ABLATED.has(idx)) continue;
      drawNode(idx, "rgba(255,255,255,0.35)", "rgba(74,111,165,0.78)", 5.2);
    }
    for (const idx of ABLATED) {
      drawNode(idx, "rgba(183,60,42,0.94)", "rgba(255,255,255,0.92)", 5.8);
    }
    if (neighbourAnchor !== null) {
      drawNode(neighbourAnchor, "rgba(74,111,165,0.92)", "rgba(255,255,255,0.95)", 6.2);
    }
  }

  function drawCommunityFeatureLabels(positions) {
    const picks = getLabelPicks("community", 8, { edgeFrom: 1, edgeTo: 1, ablated: 10, anchor: 8 });
    CTX.save();
    for (const idx of picks) {
      const p = communityPosition(positions, idx);
      if (!p) continue;
      drawFeatureLabel(idx, p[0], p[1], ABLATED.has(idx) ? "rgba(183,60,42,0.68)" : "rgba(74,111,165,0.62)");
    }
    CTX.restore();
  }

  function roundRectOn(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function setRelationshipRenderMode(mode) {
    RELATIONSHIP_RENDER_MODE = mode === "communities" ? "communities" : "links";
    const links = $("relation-links");
    const communities = $("relation-communities");
    if (links) links.classList.toggle("on", RELATIONSHIP_RENDER_MODE === "links");
    if (communities) communities.classList.toggle("on", RELATIONSHIP_RENDER_MODE === "communities");
    const suffix = relationshipModeSuffix();
    const note = $("map-note");
    if (note && MAP_NOTE_TEXT) note.textContent = MAP_NOTE_TEXT + suffix;
    requestRedraw();
  }

  function relationshipModeSuffix() {
    if (RELATIONSHIP_RENDER_MODE === "communities") {
      return " Communities view reorganizes the same atoms into precomputed relationship-cluster clouds.";
    }
    return "";
  }

  function drawFeatureLabel(idx, x, y, color) {
    const f = LAYOUT.features[idx];
    if (!f) return;
    const label = `${idx} · ${(f.label || "feature").slice(0, 34)}`;
    CTX.font = `600 ${10.5 * DPR}px -apple-system, BlinkMacSystemFont, "Inter", sans-serif`;
    const padX = 9 * DPR;
    const w = CTX.measureText(label).width + padX * 2;
    const h = 22 * DPR;
    const bx = Math.max(10 * DPR, Math.min(x + 10 * DPR, CANVAS.width - w - 10 * DPR));
    const by = Math.max(10 * DPR, Math.min(y - h - 10 * DPR, CANVAS.height - h - 10 * DPR));
    CTX.fillStyle = "rgba(255,255,255,0.92)";
    CTX.strokeStyle = color;
    CTX.lineWidth = 1 * DPR;
    roundRect(bx, by, w, h, 5 * DPR);
    CTX.fill();
    CTX.stroke();
    CTX.fillStyle = "rgba(36,35,32,0.84)";
    CTX.textBaseline = "middle";
    CTX.fillText(label, bx + padX, by + h / 2);
  }

  function drawRelationshipLabels() {
    if (!LOCAL_GRAPH && GRAPH_EDGES.length === 0 && ABLATED.size === 0) return;
    if (LOCAL_GRAPH && LOCAL_GRAPH.title === "Prompt slice") {
      CTX.save();
      for (const idx of ABLATED) {
        const f = LAYOUT.features[idx];
        if (!f) continue;
        const [x, y] = xy(f);
        if (x < -20 || x > CANVAS.width + 20 || y < -20 || y > CANVAS.height + 20) continue;
        drawFeatureLabel(idx, x, y, "rgba(183,60,42,0.68)");
      }
      CTX.restore();
      return;
    }
    const typed = new Map((LOCAL_GRAPH && LOCAL_GRAPH.nodes ? LOCAL_GRAPH.nodes : [])
      .map(n => [n.idx, n.type]));
    const picks = getLabelPicks("atlas", 5, { edgeFrom: 2, edgeTo: 1, ablated: 6, anchor: 5 });
    CTX.save();
    for (const idx of picks) {
      const f = LAYOUT.features[idx];
      if (!f) continue;
      const [x, y] = xy(f);
      if (x < -20 || x > CANVAS.width + 20 || y < -20 || y > CANVAS.height + 20) continue;
      const type = typed.get(idx);
      const color = ABLATED.has(idx) || type === "silenced"
        ? "rgba(183,60,42,0.68)"
        : type === "behavior"
          ? "rgba(198,145,22,0.72)"
          : "rgba(74,111,165,0.62)";
      drawFeatureLabel(idx, x, y, color);
    }
    CTX.restore();
  }

  function getLabelPicks(kind, limit, weights) {
    const edgeSig = GRAPH_EDGES.slice(0, 16)
      .map(e => `${e.from}-${e.to}-${Number(e.score || 0).toFixed(3)}`)
      .join("|");
    const graphSig = LOCAL_GRAPH
      ? `${LOCAL_GRAPH.title}:${LOCAL_GRAPH.nodes ? LOCAL_GRAPH.nodes.length : 0}`
      : "none";
    const key = [
      kind, limit, ablatedRenderVersion, ABLATED.size, neighbourAnchor ?? "none", edgeSig, graphSig,
    ].join(":");
    if (labelPickCache.has(key)) return labelPickCache.get(key);
    const scores = new Map();
    for (const edge of GRAPH_EDGES) {
      scores.set(edge.from, (scores.get(edge.from) || 0) + weights.edgeFrom + Number(edge.score || 0));
      scores.set(edge.to, (scores.get(edge.to) || 0) + weights.edgeTo + Number(edge.score || 0));
    }
    for (const idx of ABLATED) scores.set(idx, (scores.get(idx) || 0) + weights.ablated);
    if (neighbourAnchor !== null) scores.set(neighbourAnchor, (scores.get(neighbourAnchor) || 0) + weights.anchor);
    const picks = [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([idx]) => idx);
    labelPickCache.set(key, picks);
    if (labelPickCache.size > 12) labelPickCache.clear();
    return picks;
  }

  // ─── Local relationship graph ─────────────────────────────────────────
  function setupRelationshipCanvas() {
    REL_CANVAS = CANVAS;
    REL_CTX = CTX;
  }

  function scheduleGraphCommunityPrebuild() {
    const run = () => {
      if (!LAYOUT || !CANVAS || graphCommunityPositions) return;
      ensureGraphCommunityLayout(CANVAS.width, CANVAS.height);
    };
    if ("requestIdleCallback" in window) window.requestIdleCallback(run, { timeout: 1800 });
    else setTimeout(run, 250);
  }

  function featureByIdx(idx) {
    return LAYOUT && LAYOUT.features ? LAYOUT.features[idx] : null;
  }
  function featureLabel(idx) {
    const f = featureByIdx(idx);
    return f && f.label ? f.label : "(no label)";
  }
  function setRelationshipNote(text) {
    setMapNote(text);
  }
  function showRelationshipEmpty(show) {
    const el = $("relationship-empty");
    if (el) el.style.display = show ? "flex" : "none";
  }
  function setRelationshipTitle(text) {
    const el = $("relationship-title");
    if (el) el.textContent = text;
    const hud = $("hud_step");
    if (hud) hud.textContent = text;
  }
  function localColor(type) {
    if (type === "silenced") return "rgba(183, 60, 42, 0.95)";
    if (type === "behavior") return "rgba(198, 145, 22, 0.95)";
    if (type === "linked") return "rgba(44, 138, 74, 0.95)";
    return "rgba(74, 111, 165, 0.95)";
  }
  function mergeNodeType(current, next) {
    const rank = { linked: 0, prompt: 1, behavior: 2, silenced: 3 };
    return (rank[next] || 0) > (rank[current] || 0) ? next : current;
  }
  function makeNode(idx, type = "linked", score = 0) {
    return { idx, type, score, label: featureLabel(idx), x: 0, y: 0, vx: 0, vy: 0 };
  }

  function clusterRelationshipSubgraph(nodes, edges) {
    if (!nodes.length) return [];
    if (!edges.length) {
      const byCid = new Map();
      for (const n of nodes) {
        const f = featureByIdx(n.idx);
        const cid = f && Number.isFinite(Number(f.cid)) ? Number(f.cid) : -1;
        if (!byCid.has(cid)) byCid.set(cid, []);
        byCid.get(cid).push(n.idx);
      }
      const typeById = new Map(nodes.map(n => [n.idx, n.type]));
      return [...byCid.entries()]
        .sort((a, b) => b[1].length - a[1].length)
        .slice(0, 8)
        .map(([cid, members]) => ({
          cid,
          members,
          core: [...members].sort((a, b) => {
            const ta = typeById.get(a);
            const tb = typeById.get(b);
            const pa = (ta === "silenced" || ta === "behavior") ? 1 : 0;
            const pb = (tb === "silenced" || tb === "behavior") ? 1 : 0;
            return pb - pa;
          }).slice(0, 3),
        }));
    }
    const ids = nodes.map(n => n.idx);
    const idSet = new Set(ids);
    const adj = new Map(ids.map(id => [id, new Map()]));
    for (const e of edges) {
      if (!idSet.has(e.from) || !idSet.has(e.to)) continue;
      const w = 0.001 + Math.max(0, Number(e.score || 0));
      adj.get(e.from).set(e.to, (adj.get(e.from).get(e.to) || 0) + w);
      adj.get(e.to).set(e.from, (adj.get(e.to).get(e.from) || 0) + w);
    }
    const labels = new Map(ids.map(id => [id, id]));
    for (let iter = 0; iter < 18; iter++) {
      let changed = false;
      for (const id of ids) {
        const votes = new Map();
        for (const [to, w] of adj.get(id)) {
          const label = labels.get(to);
          votes.set(label, (votes.get(label) || 0) + w);
        }
        if (!votes.size) continue;
        const best = [...votes.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];
        if (best !== labels.get(id)) {
          labels.set(id, best);
          changed = true;
        }
      }
      if (!changed) break;
    }
    const groups = new Map();
    for (const id of ids) {
      const label = labels.get(id);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(id);
    }
    const typeById = new Map(nodes.map(n => [n.idx, n.type]));
    const degree = new Map(ids.map(id => [id, 0]));
    for (const e of edges) {
      degree.set(e.from, (degree.get(e.from) || 0) + Number(e.score || 0) + 1);
      degree.set(e.to, (degree.get(e.to) || 0) + Number(e.score || 0) + 1);
    }
    return [...groups.values()]
      .filter(members => members.length >= 2 || members.some(id => typeById.get(id) === "silenced" || typeById.get(id) === "behavior"))
      .sort((a, b) => b.length - a.length)
      .slice(0, 8)
      .map(members => ({
        members,
        core: [...members].sort((a, b) => {
          const ta = typeById.get(a);
          const tb = typeById.get(b);
          const pa = (ta === "silenced" || ta === "behavior") ? 1000 : 0;
          const pb = (tb === "silenced" || tb === "behavior") ? 1000 : 0;
          return (pb + (degree.get(b) || 0)) - (pa + (degree.get(a) || 0));
        }).slice(0, 3),
      }));
  }

  async function fetchActivationEdges(seedIds, opts = {}) {
    const seedSet = new Set(seedIds);
    const nodes = new Map(seedIds.map(idx => [idx, makeNode(idx, opts.seedType || "prompt")]));
    const edges = new Map();
    const anchors = seedIds.slice(0, opts.maxAnchors || 36);
    const maxNodes = opts.maxNodes || 72;
    const includeExternal = opts.includeExternal || false;
    const k = opts.k || 12;
    const rankBy = opts.rankBy || "jaccard";
    const missing = [];
    const consume = (anchor, feats, scores) => {
      for (let i = 0; i < feats.length; i++) {
        const to = feats[i];
        const score = Number(scores[i] || 0);
        const inSeed = seedSet.has(to);
        if (!inSeed && (!includeExternal || nodes.size >= maxNodes)) continue;
        if (!nodes.has(to)) nodes.set(to, makeNode(to, "linked", score));
        const key = anchor < to ? `${anchor}:${to}` : `${to}:${anchor}`;
        const prev = edges.get(key);
        if (!prev || score > prev.score) edges.set(key, { from: anchor, to, score, kind: "coactivation" });
      }
    };
    for (const anchor of anchors) {
      const key = `${anchor}:${k}:${rankBy}`;
      const cached = GRAPH_CACHE.get(key);
      if (cached) consume(anchor, cached.features, cached.scores);
      else missing.push(anchor);
    }
    if (missing.length && SUPPORTS_GRAPH_BATCH) {
      try {
        const r = await probe({
          cmd: "graph", query: "coact_batch", anchors: missing, k, rank_by: rankBy,
        }, { timeoutMs: 12000 });
        if (r.ok && r.result && r.result.by_anchor) {
          for (const anchor of missing) {
            const row = r.result.by_anchor[String(anchor)] || { features: [], scores: [] };
            GRAPH_CACHE.set(`${anchor}:${k}:${rankBy}`, row);
            consume(anchor, row.features || [], row.scores || []);
          }
          return { nodes: [...nodes.values()], edges: [...edges.values()] };
        }
      } catch (e) {
        // Older daemons do not know the batch query; fall back below.
      }
    }
    for (const anchor of missing) {
      try {
        const r = await probe({
          cmd: "graph", query: "coact_partners", anchor, k, rank_by: rankBy,
        }, { timeoutMs: 12000 });
        if (!r.ok || !r.result) continue;
        const feats = r.result.features || [];
        const scores = r.result.scores || [];
        GRAPH_CACHE.set(`${anchor}:${k}:${rankBy}`, { features: feats, scores });
        consume(anchor, feats, scores);
      } catch (e) {
        // Keep the local graph useful even if one anchor has no graph edges.
      }
    }
    return { nodes: [...nodes.values()], edges: [...edges.values()] };
  }

  function layoutLocalGraph(nodes, edges) {
    const W = REL_CANVAS.width, H = REL_CANVAS.height;
    const cx = W / 2, cy = H / 2;
    const radius = Math.min(W, H) * 0.32;
    nodes.forEach((n, i) => {
      const a = (i / Math.max(1, nodes.length)) * Math.PI * 2;
      n.x = cx + Math.cos(a) * radius * (0.65 + (i % 5) * 0.06);
      n.y = cy + Math.sin(a) * radius * (0.65 + (i % 7) * 0.045);
      n.vx = 0; n.vy = 0;
    });
    const byIdx = new Map(nodes.map(n => [n.idx, n]));
    for (let iter = 0; iter < 180; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y;
          let d2 = dx * dx + dy * dy + 0.01;
          const force = (2600 * DPR * DPR) / d2;
          const d = Math.sqrt(d2);
          dx /= d; dy /= d;
          a.vx += dx * force; a.vy += dy * force;
          b.vx -= dx * force; b.vy -= dy * force;
        }
      }
      for (const e of edges) {
        const a = byIdx.get(e.from), b = byIdx.get(e.to);
        if (!a || !b) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const target = (82 - Math.min(45, e.score * 70)) * DPR;
        const force = (d - target) * 0.018;
        const fx = (dx / d) * force, fy = (dy / d) * force;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
      for (const n of nodes) {
        n.vx += (cx - n.x) * 0.004;
        n.vy += (cy - n.y) * 0.004;
        n.vx *= 0.82; n.vy *= 0.82;
        n.x = Math.max(24 * DPR, Math.min(W - 24 * DPR, n.x + n.vx));
        n.y = Math.max(24 * DPR, Math.min(H - 24 * DPR, n.y + n.vy));
      }
    }
  }

  function drawLocalGraph() {
    if (!REL_CTX || !REL_CANVAS) return;
    const W = REL_CANVAS.width, H = REL_CANVAS.height;
    REL_CTX.clearRect(0, 0, W, H);
    if (!LOCAL_GRAPH || LOCAL_GRAPH.nodes.length === 0) {
      showRelationshipEmpty(true);
      REL_CTX.save();
      REL_CTX.font = `${14 * DPR}px -apple-system, BlinkMacSystemFont, sans-serif`;
      REL_CTX.fillStyle = "rgba(112, 109, 101, 0.9)";
      REL_CTX.textAlign = "center";
      REL_CTX.fillText("Choose a relationship slice to inspect.", W / 2, H / 2);
      REL_CTX.restore();
      return;
    }
    showRelationshipEmpty(false);
    const byIdx = new Map(LOCAL_GRAPH.nodes.map(n => [n.idx, n]));
    REL_CTX.save();
    REL_CTX.lineCap = "round";
    for (const e of LOCAL_GRAPH.edges) {
      const a = byIdx.get(e.from), b = byIdx.get(e.to);
      if (!a || !b) continue;
      const score = Math.max(0, Math.min(1, Number(e.score || 0)));
      REL_CTX.strokeStyle = `rgba(74, 111, 165, ${0.16 + score * 0.52})`;
      REL_CTX.lineWidth = (0.8 + score * 3.2) * DPR;
      REL_CTX.beginPath();
      REL_CTX.moveTo(a.x, a.y);
      REL_CTX.lineTo(b.x, b.y);
      REL_CTX.stroke();
    }
    for (const n of LOCAL_GRAPH.nodes) {
      const r = (n.type === "silenced" ? 6.5 : n.type === "behavior" ? 5.8 : 5.2) * DPR;
      REL_CTX.fillStyle = localColor(n.type);
      REL_CTX.beginPath();
      REL_CTX.arc(n.x, n.y, r, 0, Math.PI * 2);
      REL_CTX.fill();
      REL_CTX.strokeStyle = "rgba(255,255,255,0.92)";
      REL_CTX.lineWidth = 1.4 * DPR;
      REL_CTX.stroke();
    }
    const labelNodes = LOCAL_GRAPH.nodes
      .filter(n => n.type === "silenced" || n.type === "behavior")
      .slice(0, 18);
    REL_CTX.font = `${10.5 * DPR}px -apple-system, BlinkMacSystemFont, sans-serif`;
    REL_CTX.fillStyle = "rgba(36, 35, 32, 0.82)";
    for (const n of labelNodes) {
      REL_CTX.fillText(String(n.idx), n.x + 8 * DPR, n.y - 7 * DPR);
    }
    REL_CTX.restore();
  }

  function renderRelationshipList(nodes) {
    const list = $("relationship-list");
    if (!list) return;
    list.innerHTML = nodes.slice(0, 36).map(n => `
      <div class="rel-row">
        <span class="idx">${n.idx}</span>
        <span class="lbl">${esc(n.label || featureLabel(n.idx)).slice(0, 58)}</span>
        <span class="tag">${esc(n.type)}</span>
      </div>`).join("");
  }

  async function renderLocalGraph(title, seedNodes, opts = {}) {
    if (!REL_CANVAS || !seedNodes || seedNodes.length === 0) return;
    const run = ++localGraphRun;
    setRelationshipTitle(title);
    setRelationshipNote(opts.loading || "Building local activation graph…");
    showRelationshipEmpty(false);
    const seedIds = [...new Set(seedNodes.map(n => n.idx))].slice(0, opts.maxSeedNodes || 60);
    const typed = new Map();
    for (const n of seedNodes) {
      if (!typed.has(n.idx)) typed.set(n.idx, makeNode(n.idx, n.type, n.score || 0));
      else typed.get(n.idx).type = mergeNodeType(typed.get(n.idx).type, n.type);
    }
    let built = { nodes: [...typed.values()], edges: opts.edges || [] };

    // Show the relationship-community view immediately from served feature
    // metadata. Neo4j edge evidence can arrive later without blocking the UI.
    LOCAL_GRAPH = { title, nodes: built.nodes, edges: opts.edges || [], allEdges: opts.edges || [] };
    GRAPH_EDGES = opts.edges || [];
    MATCHED = new Set(built.nodes.map(n => n.idx));
    RELATIONSHIP_COMMUNITIES = clusterRelationshipSubgraph(built.nodes, opts.edges || []);
    const immediateAnchor = built.nodes.find(n => n.type === "silenced" || n.type === "behavior") || built.nodes[0];
    neighbourAnchor = immediateAnchor ? immediateAnchor.idx : null;
    neighbourPulse = neighbourAnchor === null ? 0 : 1;
    renderRelationshipList(built.nodes);
    setRelationshipTitle(title);
    setRelationshipNote(opts.note || "Relationship communities are shown from served graph-community metadata; fetching link evidence in the background.");
    redraw();

    if (!opts.edges) {
      built = await fetchActivationEdges(seedIds, {
        seedType: opts.seedType || "prompt",
        includeExternal: !!opts.includeExternal,
        maxNodes: opts.maxNodes || 72,
        maxAnchors: opts.maxAnchors || 36,
        k: opts.k || 12,
      });
      for (const n of built.nodes) {
        const t = typed.get(n.idx);
        if (t) n.type = mergeNodeType(n.type, t.type);
      }
    }
    if (run !== localGraphRun) return;
    const sortedEdges = [...built.edges].sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
    const maxEdges = Number.isFinite(opts.maxEdges) ? opts.maxEdges : 12;
    const visibleEdges = opts.showEdges === false ? [] : sortedEdges.slice(0, maxEdges);
    RELATIONSHIP_COMMUNITIES = clusterRelationshipSubgraph(built.nodes, sortedEdges);
    LOCAL_GRAPH = { title, nodes: built.nodes, edges: visibleEdges, allEdges: sortedEdges };
    GRAPH_EDGES = visibleEdges;
    MATCHED = new Set(built.nodes.map(n => n.idx));
    const anchor = built.nodes.find(n => n.type === "silenced" || n.type === "behavior") || built.nodes[0];
    neighbourAnchor = anchor ? anchor.idx : null;
    neighbourPulse = neighbourAnchor === null ? 0 : 1;
    renderRelationshipList(built.nodes);
    const edgeText = visibleEdges.length === 0
      ? "No co-activation edges were returned inside this slice yet."
      : `${visibleEdges.length} strongest co-activation edge${visibleEdges.length === 1 ? "" : "s"} among ${built.nodes.length} feature atoms.`;
    const communityText = RELATIONSHIP_COMMUNITIES.length
      ? ` ${RELATIONSHIP_COMMUNITIES.length} graph communit${RELATIONSHIP_COMMUNITIES.length === 1 ? "y" : "ies"} found in this relevant subgraph.`
      : "";
    setRelationshipNote(opts.note || `${edgeText}${communityText} Screen distance still comes from the 2D decoder projection.`);
    setRelationshipTitle(title);
    redraw();
  }

  function renderPromptSliceGraph() {
    if (!LAST_PROMPT_SLICE) {
      LOCAL_GRAPH = null;
      RELATIONSHIP_COMMUNITIES = [];
      setRelationshipTitle("Prompt slice");
      setRelationshipNote("Run Smart silence or Analyze prompt first.");
      return;
    }
    const silenced = new Set(LAST_PROMPT_SLICE.intersection || []);
    const nodes = [];
    for (const idx of (LAST_PROMPT_SLICE.retrieved || []).slice(0, 52)) {
      nodes.push({ idx, type: silenced.has(idx) ? "silenced" : "prompt" });
    }
    for (const idx of silenced) nodes.push({ idx, type: "silenced" });
    const unique = [...new Map(nodes.map(n => [n.idx, makeNode(n.idx, n.type)])).values()];
    LOCAL_GRAPH = { title: "Prompt slice", nodes: unique, edges: [], allEdges: [] };
    GRAPH_EDGES = [];
    MATCHED = new Set(unique.map(n => n.idx));
    RELATIONSHIP_COMMUNITIES = clusterRelationshipSubgraph(unique, []);
    neighbourAnchor = silenced.size ? [...silenced][0] : (unique[0] && unique[0].idx);
    neighbourPulse = neighbourAnchor === null || neighbourAnchor === undefined ? 0 : 1;
    setRelationshipTitle("Prompt slice");
    setRelationshipNote("Prompt-touched atoms are highlighted in blue. The intervention is the red overlap: the features actually silenced for this prompt.");
    redraw();
  }

  function renderAiismGraph() {
    if (!TOP25 || TOP25.length === 0) {
      LOCAL_GRAPH = null;
      RELATIONSHIP_COMMUNITIES = [];
      setRelationshipTitle(CONSTRUCTION_NAME);
      setRelationshipNote("The construction feature set is still loading.");
      return;
    }
    renderLocalGraph(CONSTRUCTION_NAME, TOP25.map(idx => ({ idx, type: "behavior" })), {
      includeExternal: true,
      maxNodes: 42,
      maxSeedNodes: 25,
      maxAnchors: 8,
      k: 4,
      maxEdges: 10,
      note: "The construction features stay in atlas space. A few strongest relationship edges are shown, not the whole graph.",
    });
  }

  function renderSelectedLinksGraph() {
    if (!LAST_SELECTED_LINKS) {
      LOCAL_GRAPH = null;
      RELATIONSHIP_COMMUNITIES = [];
      setRelationshipTitle("Selected feature links");
      setRelationshipNote("Option-click a feature on the map first.");
      return;
    }
    renderLocalGraph(LAST_SELECTED_LINKS.title, LAST_SELECTED_LINKS.nodes, {
      edges: LAST_SELECTED_LINKS.edges,
      maxEdges: 8,
      note: LAST_SELECTED_LINKS.note,
    });
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
    const communityLayout = RELATIONSHIP_RENDER_MODE === "communities"
      ? ensureGraphCommunityLayout(CANVAS.width, CANVAS.height)
      : null;
    const communityTransform = communityLayout ? communityViewTransform(CANVAS.width, CANVAS.height) : null;
    for (let i = 0; i < feats.length; i++) {
      const p = communityLayout
        ? (communityLayout.positions[i] ? transformCommunityPoint(communityLayout.positions[i], communityTransform) : null)
        : xy(feats[i]);
      if (!p) continue;
      const [x, y] = p;
      if (pointInPolygon(x, y, poly)) hits.push(i);
    }
    return hits;
  }

  // "What concepts does this prompt touch?" — RAG-for-activations.
  // Embeds the prompt server-side, vector-searches the 16k feature labels,
  // highlights matches on the cloud, surfaces them in search-info. User
  // decides whether to silence by clicking "Silence".
  async function retrieveConcepts() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return;
    const btn = $("retrieve-btn");
    const restore = setBusy(btn, "Finding concepts…");
    $("search").value = "";
    $("search-info").innerHTML = `Embedding the prompt and searching 16,384 feature labels…`;
    setMapNote("Prompt analysis: blue rings are concepts touched by the prompt. They can be far apart because one prompt spans several feature groups.");
    try {
      const r = await probe({
        cmd: "concept_retrieve", prompt, k: 15, expand_neighbours: 0,
      });
      if (!r.ok) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">error: ${esc(r.error || "unknown")}</span>`;
        return;
      }
      const res = r.result;
      MATCHED = new Set(res.features);
      GRAPH_EDGES = [];
      RELATIONSHIP_COMMUNITIES = [];
      neighbourAnchor = null; neighbourPulse = 0;
      LAST_PROMPT_SLICE = { retrieved: res.features || [], intersection: [], matches: res.matches || [] };
      setMapNote("Prompt analysis: blue rings are prompt-touched candidates, not graph links. Distance is the 2D projection, not the full relationship.");
      const top = res.matches.slice(0, 5).map(m =>
        `<span style="display: inline-block; background: var(--neo-soft); color: var(--neo); padding: 1px 6px; border-radius: 3px; margin: 2px 3px 0 0; font-size: 11px;">${esc(m.label.slice(0,42))}</span>`
      ).join("");
      $("search-info").innerHTML = `<strong>${res.matches.length}</strong> features match this prompt<div style="margin-top: 4px;">${top}</div><div style="font-size: 11px; color: var(--faint); margin-top: 4px;">Click <em>Silence</em> to remove these matching features, then generate again.</div>`;
      $("search").value = `concepts in: ${prompt.slice(0, 30)}…`;
      renderPromptSliceGraph();
      startAnim();
      setTimeout(stopAnim, 1500);
    } catch (e) {
      $("search-info").innerHTML = `<span style="color: var(--oppose);">network: ${describeError(e)}</span>`;
    } finally {
      restore();
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
    const restore = setBusy(btn, "Composing…");
    try {
      const r = await probe({ cmd: "compose_behaviours", intensities });
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
      $("mixer-info").innerHTML = `<strong>${res.n_total} features composed</strong> · ${parts}`;
      redraw();
      // Now auto-generate
      btn.textContent = "Generating…";
      await generate();
    } catch (e) {
      $("mixer-info").innerHTML = `<span style="color: var(--oppose);">network: ${describeError(e)}</span>`;
    } finally {
      restore();
    }
  }

  // ── Demo 1: Smart silence (concept retrieval ∩ behaviour coalition) ──
  // Asks the daemon to intersect prompt-relevant concepts with the construction
  // coalition stored as a :Behaviour node in Neo4j. Silences only the
  // overlap — i.e. only the construction features the prompt would have used.
  // This is graph set algebra in product form.
  async function surgicalDeslop() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return;
    const btn = $("surgical-btn");
    const restore = setBusy(btn, "Finding overlap…");
    $("search").value = "";
    $("search-info").innerHTML = `Checking which not-this-but-that features this prompt is likely to use…`;
    setSilenceInfo("Checking the prompt against the not-this-but-that feature set…");
    setMapNote("Smart silence: blue rings are prompt-matched features; red dots are the overlap actually silenced.");
    try {
      const r = await probe({
        cmd: "surgical_deslop", prompt, behaviour: "ai-ism", k: 80,
      });
      if (!r.ok) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">error: ${esc(r.error || "unknown")}</span>`;
        setSilenceInfo(`<span style="color: var(--oppose);">Smart silence could not finish: ${esc(r.error || "unknown")}</span>`);
        return;
      }
      const res = r.result;
      MATCHED = new Set(res.retrieved_features);
      GRAPH_EDGES = [];
      RELATIONSHIP_COMMUNITIES = [];
      neighbourAnchor = null; neighbourPulse = 0;
      LAST_PROMPT_SLICE = {
        retrieved: res.retrieved_features || [],
        intersection: res.intersection_features || [],
        matches: res.matches || [],
      };
      setMapNote("Smart silence: blue rings show prompt-matched features; red dots show the not-this-but-that features being silenced.");

      // The intersection replaces the current ablation set. Even a zero-sized
      // overlap is meaningful: Smart silence decided this prompt does not need
      // the construction removed.
      ABLATED = new Set(res.intersection_features || []);
      ABLATION_SOURCES.clear();
      document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("on"));
      btn.classList.add("on");
      if (res.intersection_features.length > 0) {
        recordSource(res.intersection_features, "surgical_deslop",
          `prompt-aware: ${prompt.slice(0, 32)}… ∩ ${CONSTRUCTION_NAME}`);
      }
      renderAblated();

      // Pretty UI: show both the matches AND the Cypher
      const matchChips = res.matches.map(m =>
        `<span style="display: inline-block; background: rgba(183,60,42,0.12); color: var(--oppose); padding: 1px 7px; border-radius: 3px; margin: 2px 3px 0 0; font-size: 11px;">feat ${m.idx} · ${esc(m.label.slice(0,32))}</span>`
      ).join("");

      if (res.intersection_features.length === 0) {
        setSilenceInfo("<strong>0</strong> not-this-but-that features added. This prompt did not strongly touch the construction, so Smart silence left the set empty.");
        $("search-info").innerHTML = `
          <strong>No not-this-but-that features matched this prompt.</strong>
          <div style="margin-top: 4px; color: var(--faint);">Smart silence left the model alone because the prompt did not strongly touch the contrastive sentence pattern.</div>
          <details style="margin-top: 8px; font-size: 11px;">
            <summary style="color: var(--muted); cursor: pointer;">show query</summary>
            <pre style="margin: 4px 0 0; font-family: var(--mono); font-size: 11px; color: var(--muted); background: var(--bg); padding: 6px 8px; border-radius: 3px; white-space: pre-wrap;">${esc(res.cypher)}</pre>
          </details>`;
      } else {
        setSilenceInfo(`<strong>${res.n_intersection}</strong> prompt-relevant not-this-but-that features added to Currently silenced. The fixed set would remove all ${TOP25 ? TOP25.length : 25}.`);
        $("search-info").innerHTML = `
          <strong>${res.n_intersection}</strong> not-this-but-that features silenced <span style="color: var(--faint);">from ${res.n_retrieved} prompt matches</span>
          <div style="margin-top: 4px;">${matchChips}</div>
          <div style="margin-top: 6px; font-size: 11px; color: var(--faint);">These are the pattern features this prompt was likely to use. Click <strong>Generate</strong> to compare.</div>
          <details style="margin-top: 8px; font-size: 11px;">
            <summary style="color: var(--muted); cursor: pointer;">show query</summary>
            <pre style="margin: 4px 0 0; font-family: var(--mono); font-size: 11px; color: var(--muted); background: var(--bg); padding: 6px 8px; border-radius: 3px; white-space: pre-wrap;">${esc(res.cypher)}</pre>
          </details>`;
      }
      $("search").value = `surgical: ${prompt.slice(0, 30)}…`;
      renderPromptSliceGraph();
      startAnim();
      setTimeout(stopAnim, 1500);
    } catch (e) {
      $("search-info").innerHTML = `<span style="color: var(--oppose);">network: ${describeError(e)}</span>`;
      setSilenceInfo(`<span style="color: var(--oppose);">Smart silence could not reach the daemon: ${describeError(e)}</span>`);
    } finally {
      restore();
    }
  }

  // Alt-click: show this feature's top relationship-linked features from Neo4j.
  // Immediate visual feedback (pulse on anchor) before the response lands; then
  // highlight linked features BIG and BLUE so it's unmistakable.
  async function showLinkedFeatures(idx) {
    const advanced = document.querySelector(".pg-mixer");
    if (advanced) advanced.open = true;
    // 1. Immediate feedback — pulse the clicked feature so the user sees the gesture registered
    neighbourAnchor = idx;
    neighbourPulse = 1.0;
    const lbl = LAYOUT.features[idx].label || "(no label)";
    $("search-info").innerHTML = `Finding related features for feat ${idx}…`;
    $("search").value = `relationships for ${idx}`;
    setMapNote("Relationships: the lines show features that fire together in the graph. Screen distance is only the 2D UMAP projection.");
    startAnim();
    try {
      let relation = "activation";
      let scoreName = "Jaccard";
      let r = await probe({
        cmd: "graph", query: "coact_partners", anchor: idx, k: 10, rank_by: "jaccard",
      });
      if (!r.ok || !r.result || !r.result.features || r.result.features.length === 0) {
        relation = "decoder";
        scoreName = "cosine";
        r = await probe({ cmd: "graph", query: "decoder_neighbors", anchor: idx, k: 10 });
      }
      if (!r.ok) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">error fetching linked features: ${esc(r.error || "unknown")}</span>`;
        return;
      }
      const feats = r.result.features;
      MATCHED = new Set(feats);
      MATCHED.add(idx);
      GRAPH_EDGES = feats.map((to, i) => ({
        from: idx,
        to,
        score: (r.result.scores || [])[i] || 0,
      }));
      LAST_SELECTED_LINKS = {
        title: `Selected feature ${idx}`,
        nodes: [
          { idx, type: "behavior" },
          ...feats.map(to => ({ idx: to, type: relation === "activation" ? "linked" : "prompt" })),
        ],
        edges: GRAPH_EDGES,
        note: relation === "activation"
          ? "Green nodes are the selected feature's strongest co-activating partners. Lines are CO_ACTIVATES_WITH edges."
          : "This feature had no co-activation edges, so this local graph uses decoder-similarity fallback edges.",
      };
      neighbourPulse = 1.0;  // re-pulse on data arrival
      if (relation === "activation") {
        setMapNote("Relationships: blue lines are CO_ACTIVATES_WITH graph edges. Far-apart dots can still be behaviorally related.");
      } else {
        setMapNote("Decoder fallback: blue lines use decoder-vector similarity because no co-activation edges were returned for this feature.");
      }
      const relationLabel = relation === "activation" ? "co-activating" : "decoder-similar";
      $("search-info").innerHTML = `<strong>${feats.length}</strong> ${relationLabel} features for feat ${idx} (<em>${esc(lbl).slice(0,40)}</em>) — connected to the anchor with blue lines by ${scoreName}.<div style="margin-top: 4px; font-size: 11px; color: var(--faint);">Click <em>Silence</em> to remove this related set.</div>`;
      renderSelectedLinksGraph();
      startAnim();
      setTimeout(stopAnim, 1500);
    } catch (e) {
      $("search-info").innerHTML = `<span style="color: var(--oppose);">network error: ${describeError(e)}</span>`;
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
    MATCHED.clear();
    GRAPH_EDGES = [];
    RELATIONSHIP_COMMUNITIES = [];
    neighbourAnchor = null;
    neighbourPulse = 0;
    setMapNote();
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
    document.querySelectorAll(".preset-card").forEach(c => c.classList.remove("on"));
    renderAblated();
    redraw();
  }
  function communityMembers(cid) {
    if (!LAYOUT || !LAYOUT.features) return [];
    return LAYOUT.features
      .filter(f => f.cid === cid)
      .sort((a, b) => b.density - a.density)
      .map(f => f.idx);
  }
  function clearAblation() {
    ABLATED = new Set();
    MATCHED.clear();
    GRAPH_EDGES = [];
    RELATIONSHIP_COMMUNITIES = [];
    neighbourAnchor = null;
    neighbourPulse = 0;
    setMapNote();
    ABLATION_SOURCES.clear();
    document.querySelectorAll(".preset-card").forEach(c => c.classList.toggle("on", c.dataset.preset === "none"));
    const search = $("search");
    if (search) search.value = "";
    const info = $("search-info");
    if (info) info.textContent = "Search highlights matching features. Click Silence to add those matches to Currently silenced.";
    setSilenceInfo("Smart silence is usually the useful path: it finds the prompt-specific overlap, then adds those features to Currently silenced.");
    setMapMode("atlas");
    renderAblated();
    redraw();
  }
  function renderAblated() {
    ablatedRenderVersion++;
    labelPickCache.clear();
    $("ablated-count").textContent = ABLATED.size;
    const list = $("ablated-list");
    if (ABLATED.size === 0) {
      list.innerHTML = `<div class="empty-note">Nothing silenced yet. Click Smart silence, a feature group, a preset, or any dot on the graph.</div>`;
      return;
    }
    const items = [...ABLATED].sort((a, b) => a - b);
    const visible = items.slice(0, 240);
    const hidden = items.length - visible.length;
    list.innerHTML = visible.map(i => {
      const f = LAYOUT.features[i];
      const lbl = (f.label || "(no label)").slice(0, 32);
      return `<div class="feat-row ablated" data-idx="${i}">
        <span class="idx">${i}</span>
        <span class="lbl">${esc(lbl)}</span>
        <span class="x" data-rm="${i}">×</span>
      </div>`;
    }).join("") + (hidden > 0
      ? `<div class="empty-note ablated-overflow">Showing 240 of ${items.length.toLocaleString()} silenced features. Use Clear all to reset the set.</div>`
      : "");
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
    GRAPH_EDGES = [];
    RELATIONSHIP_COMMUNITIES = [];
    neighbourAnchor = null;
    neighbourPulse = 0;
    setMapNote();
    if (!q) {
      $("search-info").textContent = "Type to filter features by their label.";
      redraw(); return;
    }
    setMapNote("Concept search: blue rings are label matches. They are a filter overlay, not a claim that the dots are adjacent in the graph.");
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
    if (name === "top25") {
      if (!TOP25 || TOP25.length === 0) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">The construction feature set is still loading. Try again in a moment.</span>`;
        setSilenceInfo(`<span style="color: var(--oppose);">The fixed not-this-but-that set is still loading. Try again in a moment.</span>`);
        return;
      }
      setAblation(TOP25, { kind: "preset", label: `preset: ${CONSTRUCTION_NAME} (top-25)` });
      setSilenceInfo(`<strong>${TOP25.length}</strong> fixed not-this-but-that features added. This skips prompt analysis and removes the whole construction set.`);
      renderAiismGraph();
      return;
    }
    if (name === "minimal")      return setAblation(MINIMAL,             { kind: "preset", label: "preset: minimal core (3223+9909)" });
    if (name === "suppressors") {
      if (!SUPPRESSORS10 || SUPPRESSORS10.length === 0) {
        $("search-info").innerHTML = `<span style="color: var(--oppose);">The stress-test feature set is still loading. Try again in a moment.</span>`;
        return;
      }
      setAblation(SUPPRESSORS10, { kind: "stress_test", label: "stress test: amplify not-this-but-that" });
      setSilenceInfo(`<strong>${SUPPRESSORS10.length}</strong> stress-test features added from Advanced controls.`);
      setMapNote("Stress test selected: these silenced features usually dampen the pattern, so this can make not-this-but-that easier to see.");
      return;
    }
  }

  // ─── community navigator ───────────────────────────────────────────────
  // Lists communities sorted by size. Each row hover → highlights the region
  // on the map; click → silences the whole group.
  function renderCommunityNav() {
    const nav = $("comm-nav");
    if (!COMM) { nav.innerHTML = "<div style='color: var(--faint); font-size: 11px;'>(community names not loaded)</div>"; return; }
    const entries = Object.values(COMM).sort((a, b) => b.size - a.size);
    nav.innerHTML = entries.map(e => {
      // Build a colour swatch using the same hue function as the map
      // We can't call commColor before the canvas is ready; use a CSS
      // variable that the redraw will style later — but it's simpler to
      // just compute the inline style with the same golden-angle hue logic.
      const tone = communityTone(e.cid);
      return `
      <div class="comm-item" data-cid="${e.cid}" title="Click to silence this whole group. ${esc(e.exemplar_labels.slice(0,3).join(' · '))}">
        <span class="swatch" style="background: hsla(${tone.h}, ${tone.s}%, ${Math.min(64, tone.l + 8)}%, 0.62)"></span>
        <span class="nm">${esc(e.name)}</span>
        <span class="sz">${e.size}</span>
      </div>`;
    }).join("");
    nav.querySelectorAll(".comm-item").forEach(item => {
      const cid = parseInt(item.dataset.cid, 10);
      item.addEventListener("mouseenter", () => {
        hoveredCid = cid;
        baseKey = "";
        requestRedraw();
      });
      item.addEventListener("mouseleave", () => {
        hoveredCid = null;
        baseKey = "";
        requestRedraw();
      });
      item.addEventListener("click", () => {
        const members = communityMembers(cid);
        if (!members.length) return;
        const name = (COMM[cid] && COMM[cid].name) || `cid ${cid}`;
        addToAblation(members, { kind: "community_click",
          label: `community: ${name} (all ${members.length.toLocaleString()} features)` });
        setMapNote(`Silenced the whole "${name}" feature group (${members.length.toLocaleString()} features). Output only changes if those features would fire for this prompt.`);
      });
    });
  }

  // ─── generate ──────────────────────────────────────────────────────────
  function finishActiveRun(run) {
    if (activeRun !== run) return;
    if (statusTimer) {
      clearInterval(statusTimer);
      statusTimer = null;
    }
    $("cancel-run").hidden = true;
    activeRun = null;
  }
  function cancelActiveRun() {
    if (!activeRun) return;
    playSession += 1;
    activeRun.controller.abort("cancelled");
    finishActiveRun(activeRun);
    $("generate").disabled = false;
    $("generate").textContent = "Generate";
    $("hud_step").textContent = "cancelled";
    stopAnim();
  }

  async function generate() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return;
    if (activeRun) cancelActiveRun();
    const model = $("model").value;
    const max_new = parseInt($("max_new").value, 10);
    const seed = parseInt($("seed").value, 10);
    const speedMs = parseInt($("speed").value, 10);
    const ablate = [...ABLATED];

    playSession += 1;
    const my = playSession;
    const run = { controller: new AbortController(), startedAt: performance.now() };
    activeRun = run;
    $("generate").disabled = true;
    $("generate").textContent = "Generating…";
    $("cancel-run").hidden = false;
    $("text-left").textContent = ""; $("text-right").textContent = "";
    $("vl").textContent = ""; $("vr").textContent = "";
    $("ablated-label").firstChild.textContent = ablate.length
      ? `With ${ablate.length} feature${ablate.length === 1 ? "" : "s"} silenced  `
      : "With nothing silenced (same as baseline)  ";
    ACTIVITY.fill(0);
    ACTIVE_FEATURES.clear();
    startAnim();

    try {
      const t0 = performance.now();
      run.phase = "baseline pass";
      $("hud_step").textContent = "baseline pass…";
      statusTimer = setInterval(() => {
        if (activeRun !== run) return;
        const elapsed = Math.round((performance.now() - t0) / 1000);
        const phase = run.phase || "calling daemon";
        $("hud_step").textContent = elapsed > 10
          ? `${phase} · ${elapsed}s`
          : `${phase} · ${elapsed}s`;
      }, 1000);
      const baseR = await probe({ cmd: "generate_with_activations", model, prompt,
        ablate: null, max_new_tokens: max_new, seed, top_k_features: 10 },
        { timeoutMs: 90000, signal: run.controller.signal });
      let ablR = null;
      if (ablate.length > 0) {
        run.phase = "ablated pass";
        $("hud_step").textContent = "ablated pass…";
        ablR = await probe({ cmd: "generate_with_activations", model, prompt,
          ablate, max_new_tokens: max_new, seed, top_k_features: 10 },
          { timeoutMs: 90000, signal: run.controller.signal });
      }
      if (statusTimer) {
        clearInterval(statusTimer);
        statusTimer = null;
      }

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
        for (const f of (rec.top_features || [])) {
          ACTIVITY[f.idx] = 1.0;
          ACTIVE_FEATURES.add(f.idx);
        }
        renderActiveList(rec.top_features || []);
        $("active-count").textContent = (rec.top_features || []).length;
        await sleep(speedMs);
      }

      const baseHit = hasConstruction(leftBuf.join(""));
      const ablHit = hasConstruction(rightBuf.join(""));
      const baseTopFeatures = new Set();
      for (const rec of baseRecs) {
        for (const f of (rec.top_features || [])) baseTopFeatures.add(f.idx);
      }
      const selectedThatFired = ablate.filter(idx => baseTopFeatures.has(idx));
      $("vl").className = "verdict " + (baseHit ? "hit" : "clean");
      $("vl").textContent = baseHit ? "✓ construction present" : "· clean";
      $("vr").className = "verdict " + (ablHit ? "hit" : "clean");
      $("vr").textContent = ablHit ? "✓ construction present" : "· clean";
      const wallS = (performance.now() - t0) / 1000;
      const modelS = Number(baseR.elapsed_s || 0) + Number((ablR && ablR.elapsed_s) || 0);
      const playbackS = (maxLen * speedMs) / 1000;
      const timingParts = [];
      if (modelS > 0) timingParts.push(`model ${modelS.toFixed(1)}s`);
      if (playbackS > 0.2) timingParts.push(`replay ${playbackS.toFixed(1)}s`);
      $("hud_step").textContent = `done · ${maxLen} tokens · ${wallS.toFixed(1)}s${timingParts.length ? ` (${timingParts.join(" + ")})` : ""}`;
      if (ablate.length > 0) {
        const sameOutput = leftBuf.join("") === rightBuf.join("");
        if (sameOutput && selectedThatFired.length === 0) {
          setMapNote(`The output matched because none of the ${ablate.length.toLocaleString()} silenced features appeared among the captured top activations for this prompt. Pick a prompt-aware intervention or a group this prompt actually touches.`);
        } else {
          setMapNote(`${selectedThatFired.length.toLocaleString()} of ${ablate.length.toLocaleString()} silenced features appeared among the baseline top activations captured during this run.`);
        }
      }
      setTimeout(stopAnim, 2000);

      // ── Demo 3: render audit panel + persist to Neo4j ──
      renderAudit(prompt, leftBuf.join(""), rightBuf.join(""));
      persistAudit(prompt, leftBuf.join(""), rightBuf.join(""))
        .catch(e => console.warn("persist failed:", e));
    } catch (e) {
      if (my === playSession) $("hud_step").textContent = "ERROR: " + (e.message || "unknown");
    } finally {
      finishActiveRun(run);
      if (my === playSession) {
        $("generate").disabled = false;
        $("generate").textContent = "Generate";
      }
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
    summary += `<div style="margin-top: 4px; font-size: 12px; color: var(--faint);">Click "show Cypher" to see the graph query that recorded this intervention.</div>`;

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
      const r = await probe(payload, { timeoutMs: 15000 });
      if (r.ok) LAST_INTERVENTION_ID = r.result.intervention_id;
    } catch (e) { /* swallow — UI still has the client-side audit */ }
  }

  // ─── wiring ────────────────────────────────────────────────────────────
  function wireControls() {
    $("generate").addEventListener("click", generate);
    $("cancel-run").addEventListener("click", cancelActiveRun);
    $("prompt").addEventListener("keydown", e => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate();
    });
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
    const $relationLinks = $("relation-links");
    if ($relationLinks) $relationLinks.addEventListener("click", () => setRelationshipRenderMode("links"));
    const $relationCommunities = $("relation-communities");
    if ($relationCommunities) $relationCommunities.addEventListener("click", () => setRelationshipRenderMode("communities"));

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
    // Submit search on Enter — same as clicking "Silence"
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
