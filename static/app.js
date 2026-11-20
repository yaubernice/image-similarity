const state = {
  view: "studio",
  status: null,
  library: [],
  space: null,
  file: null,
  previewUrl: null,
  results: null,
  showCount: 6,
  searching: false,
  wasBuilding: false,
  wasEvaluating: false,
  compare: null,
  evalRows: [],
  filter: "",
};

const $ = (id) => document.getElementById(id);

function toast(message, bad = false) {
  const node = $("toast");
  node.textContent = message;
  node.classList.toggle("bad", bad);
  node.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => node.classList.remove("show"), 3200);
}

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    const message = typeof detail === "string"
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => item.msg || String(item)).join(" ")
        : "Request failed";
    throw new Error(message);
  }
  return body;
}

function setView(view) {
  state.view = view;
  for (const name of ["studio", "compare", "evaluate", "library", "method"]) {
    const section = $(`view-${name}`);
    section.hidden = name !== view;
    const tab = document.querySelector(`[data-nav="${name}"]`);
    if (name === view) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  if (view === "studio") requestAnimationFrame(renderMap);
  if (view === "compare") renderCompare();
  if (view === "evaluate") loadEvaluation();
  if (location.hash !== `#${view}`) {
    history.replaceState(null, "", `#${view}`);
  }
}

function renderStatus(status) {
  if (!status) return;
  state.status = status;
  const dot = $("status-dot");
  const label = $("status-label");
  dot.className = "dot";
  if (status.building) {
    dot.classList.add("busy");
    label.textContent = `Indexing ${Math.round(status.progress * 100)}%`;
  } else if (status.model_error) {
    dot.classList.add("bad");
    label.textContent = "Model unavailable";
  } else if (status.model_loading) {
    dot.classList.add("busy");
    label.textContent = `Loading ${status.active_model}`;
  } else if (!status.indexed) {
    dot.classList.add("warn");
    label.textContent = `${status.active_model} · no index`;
  } else if (status.stale) {
    dot.classList.add("warn");
    label.textContent = `${status.active_model} · ${status.new_count} new`;
  } else {
    dot.classList.add("ready");
    const dim = status.feature_dim ? `${status.feature_dim}-d · ` : "";
    label.textContent = `${status.active_model} · ${dim}${status.num_vectors}`;
  }
  const picker = $("model-select");
  if (picker && document.activeElement !== picker) picker.value = status.active_model;

  const maxK = Math.max(1, Math.min(12, status.num_vectors || 12));
  const range = $("k-range");
  range.max = String(maxK);
  if (state.showCount > maxK) {
    state.showCount = maxK;
    range.value = String(maxK);
    $("k-label").textContent = String(maxK);
  }

  const hint = $("search-hint");
  const button = $("search-btn");
  if (status.model_error) {
    hint.textContent = status.model_error;
  } else if (!status.indexed) {
    hint.textContent = `Build a ${status.active_model} index from the library before searching.`;
  } else if (status.stale) {
    hint.textContent = "The index is behind the library. Search still uses the last build.";
  } else if (status.built_at) {
    hint.textContent = `${status.feature_dim}-d ${status.model}. Built ${status.built_at}.`;
  } else {
    hint.textContent = "";
  }
  const ready = status.indexed && !status.model_error && state.file && !state.searching;
  button.disabled = !ready;
  button.textContent = state.searching ? "Searching…" : "Search the index";
  const anyIndexed = Object.values(status.models || {}).some((item) => item.indexed);
  $("compare-btn").disabled = !(anyIndexed && state.file && !state.searching && !status.model_error);
  if (status.feature_dim) $("query-meta").textContent = `${status.feature_dim}-d · unit length`;
  const clipReady = Boolean(status.models?.clip?.indexed);
  $("text-query").disabled = !clipReady;
  $("text-btn").disabled = !clipReady || !$("text-query").value.trim() || state.searching;
  $("text-query").placeholder = clipReady ? "a red sports car" : "Build a CLIP index to search with text";
  const evaluation = status.evaluation || {};
  if (state.wasEvaluating && !evaluation.running) loadEvaluation();
  state.wasEvaluating = Boolean(evaluation.running);
  const evalStatus = $("eval-status");
  if (evalStatus) {
    evalStatus.textContent = evaluation.running
      ? evaluation.message || "Scoring"
      : evaluation.error || "";
  }
  const evalButton = $("eval-btn");
  if (evalButton) evalButton.disabled = Boolean(evaluation.running || status.building);

  const banner = $("stale-banner");
  if (status.indexed && status.stale && !status.building) {
    const parts = [];
    if (status.new_count) parts.push(`${status.new_count} not yet indexed`);
    if (status.missing_count) parts.push(`${status.missing_count} indexed file${status.missing_count === 1 ? "" : "s"} missing from the library`);
    banner.hidden = false;
    banner.textContent = `${parts.join(" · ")}. Rebuild the index to refresh search.`;
  } else {
    banner.hidden = true;
  }

  renderBuild(status);
  if (state.wasBuilding && !status.building) {
    refreshLibrary();
    refreshSpace();
    if (status.error) toast(status.error, true);
    else toast(status.message || "Index updated");
  }
  state.wasBuilding = status.building;
}

function renderBuild(status) {
  const panel = $("build-panel");
  const show = status.building || status.error;
  panel.hidden = !show;
  $("build-btn").disabled = Boolean(status.building);
  const modelName = status.active_model || "model";
  $("build-btn").textContent = status.building ? "Building…" : `Build ${modelName} index`;
  if (!show) return;
  $("build-message").textContent = status.error || status.message || "Indexing";
  const pct = Math.round((status.progress || 0) * 100);
  $("build-pct").textContent = `${pct}%`;
  $("build-fill").style.width = `${pct}%`;
  const log = $("build-log");
  log.replaceChildren();
  (status.log || []).forEach((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    log.appendChild(item);
  });
  log.scrollTop = log.scrollHeight;
}

function renderQuery() {
  const hasFile = Boolean(state.file);
  $("drop-empty").hidden = hasFile;
  $("preview").hidden = !hasFile;
  if (hasFile) {
    $("preview-img").src = state.previewUrl;
    $("preview-img").alt = state.file.name;
    $("preview-name").textContent = state.file.name;
  }
  if (state.status) renderStatus(state.status);
}

function visibleResults() {
  if (!state.results) return [];
  return state.results.results.slice(0, state.showCount);
}

function renderResults() {
  const list = $("results");
  const empty = $("results-empty");
  const payload = state.results;
  if (!payload) {
    list.hidden = true;
    empty.hidden = false;
    $("fingerprint-block").hidden = true;
    $("results-meta").textContent = "Sorted by L2 distance";
    return;
  }
  empty.hidden = true;
  list.hidden = false;
  const shown = visibleResults();
  const seconds = payload.elapsed_ms >= 1000
    ? `${(payload.elapsed_ms / 1000).toFixed(1)} s`
    : `${payload.elapsed_ms} ms`;
  $("results-meta").textContent = `${shown.length} of ${payload.searched} · ${seconds}`;
  list.replaceChildren();
  shown.forEach((result) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "result";
    button.addEventListener("click", () => openCompare(result));

    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = String(result.rank).padStart(2, "0");

    const img = document.createElement("img");
    img.className = "thumb";
    img.src = result.url;
    img.alt = result.name;

    const copy = document.createElement("div");
    const name = document.createElement("p");
    name.className = "result-name";
    name.textContent = result.name;
    const sub = document.createElement("p");
    sub.className = "result-sub";
    const extra = result.inliers != null ? ` · ${result.inliers} inliers` : "";
    sub.textContent = `L2 ${result.distance.toFixed(3)} · cosine ${result.cosine.toFixed(3)}${extra}`;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("span");
    bar.appendChild(fill);
    copy.append(name, sub, bar);

    const score = document.createElement("div");
    score.className = "score";
    const strong = document.createElement("strong");
    strong.textContent = result.similarity.toFixed(2);
    const caption = document.createElement("span");
    caption.textContent = "Similarity";
    score.append(strong, caption);

    button.append(rank, img, copy, score);
    item.appendChild(button);
    list.appendChild(item);
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(0, Math.min(1, result.similarity)) * 100}%`;
    });
  });

  $("fingerprint-block").hidden = false;
  $("norm-label").textContent = payload.vector_norm != null ? `L2 norm ${payload.vector_norm.toFixed(3)}` : "Unit length";
  requestAnimationFrame(() => drawFingerprint($("query-fp"), payload.fingerprint));
}

function drawFingerprint(canvas, queryBins, matchBins) {
  if (!canvas) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const series = [queryBins, matchBins].filter(Boolean);
  let max = 0.0001;
  series.forEach((bins) => bins.forEach((value) => {
    max = Math.max(max, Math.abs(value));
  }));

  ctx.strokeStyle = "rgba(243, 240, 231, 0.16)";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  const pathFor = (bins) => {
    const path = new Path2D();
    bins.forEach((value, index) => {
      const x = (index / Math.max(bins.length - 1, 1)) * width;
      const y = height / 2 - (value / max) * (height * 0.42);
      if (index === 0) path.moveTo(x, y);
      else path.lineTo(x, y);
    });
    return path;
  };

  if (matchBins) {
    const fill = pathFor(matchBins);
    fill.lineTo(width, height / 2);
    fill.lineTo(0, height / 2);
    ctx.fillStyle = "rgba(231, 161, 90, 0.18)";
    ctx.fill(fill);
    ctx.strokeStyle = "#e7a15a";
    ctx.lineWidth = 1.4;
    ctx.stroke(pathFor(matchBins));
  }
  if (queryBins) {
    ctx.strokeStyle = "#e2f36b";
    ctx.lineWidth = 1.6;
    ctx.stroke(pathFor(queryBins));
  }
}

function renderSamples(images) {
  const wrap = $("samples");
  const row = $("sample-row");
  row.replaceChildren();
  if (!images.length) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  images.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample";
    button.title = sample.name;
    const img = document.createElement("img");
    img.src = sample.url;
    img.alt = sample.name;
    button.appendChild(img);
    button.addEventListener("click", () => useRemoteImage(sample.url, sample.name));
    row.appendChild(button);
  });
}

function renderLibrary() {
  const tiles = $("tiles");
  const empty = $("library-empty");
  const query = state.filter.trim().toLowerCase();
  const images = state.library.filter((image) => image.name.toLowerCase().includes(query));
  $("library-count").textContent = `${state.library.length} photograph${state.library.length === 1 ? "" : "s"}`;
  tiles.replaceChildren();
  empty.hidden = state.library.length !== 0;
  images.forEach((image) => {
    const tile = document.createElement("article");
    tile.className = "tile";

    const open = document.createElement("button");
    open.type = "button";
    open.className = "tile-open";
    const img = document.createElement("img");
    img.src = image.url;
    img.alt = image.name;
    img.loading = "lazy";
    const meta = document.createElement("span");
    meta.className = "tile-meta";
    const name = document.createElement("span");
    name.textContent = image.name;
    const stateLabel = document.createElement("span");
    stateLabel.textContent = image.indexed ? "Indexed" : "New";
    meta.append(name, stateLabel);
    open.append(img, meta);
    open.addEventListener("click", () => useRemoteImage(image.url, image.name));

    const flag = document.createElement("span");
    flag.className = image.indexed ? "flag on" : "flag";
    flag.title = image.indexed ? "In the current index" : "Not in the index yet";

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.textContent = "Remove";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      if (!remove.classList.contains("armed")) {
        document.querySelectorAll(".remove.armed").forEach((node) => {
          node.classList.remove("armed");
          node.textContent = "Remove";
        });
        remove.classList.add("armed");
        remove.textContent = "Confirm";
        return;
      }
      deleteImage(image.path);
    });

    tile.append(open, flag, remove);
    tiles.appendChild(tile);
  });
}

function stageSize() {
  const stage = $("map-stage");
  return { width: stage.clientWidth, height: stage.clientHeight };
}

function relax(points, width, height) {
  const minDist = 64;
  const placed = points.map((point) => ({
    ...point,
    px: point.x * width,
    py: point.y * height,
  }));
  for (let pass = 0; pass < 18; pass += 1) {
    for (let i = 0; i < placed.length; i += 1) {
      for (let j = i + 1; j < placed.length; j += 1) {
        const dx = placed[j].px - placed[i].px;
        const dy = placed[j].py - placed[i].py;
        const dist = Math.hypot(dx, dy) || 0.01;
        if (dist < minDist) {
          const push = (minDist - dist) / 2;
          const ux = dx / dist;
          const uy = dy / dist;
          placed[i].px -= ux * push;
          placed[i].py -= uy * push;
          placed[j].px += ux * push;
          placed[j].py += uy * push;
        }
      }
    }
  }
  placed.forEach((point) => {
    point.px = Math.min(width - 30, Math.max(30, point.px));
    point.py = Math.min(height - 30, Math.max(30, point.py));
  });
  return placed;
}

function renderMap() {
  const stage = $("map-stage");
  const pins = $("map-pins");
  const empty = $("map-empty");
  const points = state.space?.points || [];
  const explained = state.space?.explained || [];
  const variance = explained.reduce((sum, value) => sum + value, 0);
  $("map-var").textContent = variance
    ? `PCA · ${Math.round(variance * 100)}% variance`
    : "PCA";

  if (!points.length) {
    pins.replaceChildren();
    empty.hidden = false;
    const canvas = $("map-links");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  empty.hidden = true;
  const { width, height } = stageSize();
  if (!width || !height) return;
  const placed = relax(points, width, height);
  const hits = new Map(visibleResults().map((result) => [result.path, result]));
  const searching = Boolean(state.results);

  pins.replaceChildren();
  placed.forEach((point) => {
    const hit = hits.get(point.path);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "map-pin";
    if (searching && !hit) button.classList.add("is-dim");
    if (hit) button.classList.add("is-hit");
    button.style.left = `${(point.px / width) * 100}%`;
    button.style.top = `${(point.py / height) * 100}%`;
    button.title = hit
      ? `${hit.rank}. ${point.name} · similarity ${hit.similarity.toFixed(2)}`
      : point.name;
    const img = document.createElement("img");
    img.src = point.url;
    img.alt = point.name;
    button.appendChild(img);
    if (hit) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = String(hit.rank);
      button.appendChild(badge);
    }
    button.addEventListener("click", () => useRemoteImage(point.url, point.name));
    pins.appendChild(button);
  });

  if (state.results?.query_point) {
    const query = state.results.query_point;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "map-pin is-query";
    button.style.left = `${Math.min(94, Math.max(6, query.x * 100))}%`;
    button.style.top = `${Math.min(94, Math.max(6, query.y * 100))}%`;
    button.title = state.results.query_name || "Query";
    const img = document.createElement("img");
    img.src = state.results.query_url;
    img.alt = "Query";
    button.appendChild(img);
    pins.appendChild(button);
  }

  drawLinks();
}

function drawLinks() {
  const stage = $("map-stage");
  const canvas = $("map-links");
  const rect = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const query = stage.querySelector(".is-query");
  if (!query) return;
  const q = centerOf(query, rect);
  ctx.strokeStyle = "rgba(226, 243, 107, 0.85)";
  ctx.lineWidth = 1.25;
  stage.querySelectorAll(".is-hit").forEach((pin) => {
    const point = centerOf(pin, rect);
    ctx.beginPath();
    ctx.moveTo(q.x, q.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
  });
}

function centerOf(node, rect) {
  const box = node.getBoundingClientRect();
  return {
    x: box.left + box.width / 2 - rect.left,
    y: box.top + box.height / 2 - rect.top,
  };
}

function openCompare(result) {
  const payload = state.results;
  const body = $("compare-body");
  body.replaceChildren();

  const top = document.createElement("div");
  top.className = "compare-top";
  const title = document.createElement("div");
  title.className = "kicker";
  const left = document.createElement("span");
  left.textContent = `Rank ${String(result.rank).padStart(2, "0")}`;
  const right = document.createElement("span");
  right.textContent = result.name;
  title.append(left, right);
  const close = document.createElement("button");
  close.type = "button";
  close.className = "btn btn-ghost";
  close.textContent = "Close";
  close.addEventListener("click", () => $("compare").close());
  top.append(title, close);

  const grid = document.createElement("div");
  grid.className = "compare-grid";
  grid.append(
    labeledImage(payload.query_url, payload.query_name || "Query", "Query"),
    metricBlock(result),
    labeledImage(result.url, result.name, "Neighbour"),
  );

  const canvas = document.createElement("canvas");
  canvas.className = "fp";
  const note = document.createElement("p");
  note.className = "fine";
  note.textContent = "Lime is the query descriptor. Amber is this neighbour. Closer shapes mean a closer direction in feature space.";

  body.append(top, grid, canvas, note);
  $("compare").showModal();
  requestAnimationFrame(() => {
    drawFingerprint(canvas, payload.fingerprint, result.fingerprint);
    const ring = body.querySelector(".value");
    if (ring) {
      const length = Number(ring.dataset.length);
      ring.style.strokeDashoffset = String(length * (1 - result.similarity));
    }
  });
}

function labeledImage(url, name, caption) {
  const figure = document.createElement("figure");
  figure.style.margin = "0";
  const img = document.createElement("img");
  img.src = url;
  img.alt = name;
  const cap = document.createElement("figcaption");
  cap.className = "fine";
  cap.textContent = caption;
  figure.append(img, cap);
  return figure;
}

function metricBlock(result) {
  const wrap = document.createElement("div");
  wrap.className = "compare-mid";
  const big = document.createElement("p");
  big.className = "big";
  big.textContent = result.similarity.toFixed(2);
  const sim = document.createElement("p");
  sim.textContent = "Similarity · 1 / (1 + distance)";
  const dist = document.createElement("p");
  dist.textContent = `L2 distance ${result.distance.toFixed(4)}`;
  const cosine = document.createElement("p");
  cosine.textContent = `Cosine ${result.cosine.toFixed(4)}`;
  const radius = 28;
  const length = 2 * Math.PI * radius;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 72 72");
  svg.setAttribute("class", "meter");
  const track = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  track.setAttribute("class", "track");
  track.setAttribute("cx", "36");
  track.setAttribute("cy", "36");
  track.setAttribute("r", String(radius));
  const value = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  value.setAttribute("class", "value");
  value.setAttribute("cx", "36");
  value.setAttribute("cy", "36");
  value.setAttribute("r", String(radius));
  value.setAttribute("stroke-dasharray", String(length));
  value.setAttribute("stroke-dashoffset", String(length));
  value.dataset.length = String(length);
  svg.append(track, value);
  wrap.append(big, sim, dist, cosine, svg);
  return wrap;
}

async function refreshStatus() {
  try {
    const status = await readJson(await fetch("/api/status"));
    renderStatus(status);
  } catch (error) {
    $("status-label").textContent = "Studio offline";
    $("status-dot").className = "dot bad";
  }
}

async function refreshLibrary() {
  const data = await readJson(await fetch("/api/library"));
  state.library = data.images;
  renderLibrary();
}

async function refreshSpace() {
  state.space = await readJson(await fetch("/api/space"));
  renderMap();
}

async function refreshSamples() {
  const data = await readJson(await fetch("/api/samples"));
  renderSamples(data.images);
}

function setFile(file) {
  if (!file) return;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  state.results = null;
  renderQuery();
  renderResults();
  renderMap();
  setView("studio");
}

async function useRemoteImage(url, name) {
  const blob = await fetch(url).then((response) => response.blob());
  const type = blob.type || "image/jpeg";
  setFile(new File([blob], name, { type }));
  if (state.status?.indexed && state.status?.model_loaded) {
    doSearch();
  }
}

async function doSearch() {
  if (!state.file || state.searching) return;
  state.searching = true;
  renderQuery();
  const body = new FormData();
  body.append("file", state.file);
  body.append("top_k", "12");
  body.append("model", state.status?.active_model || "vgg19");
  body.append("aqe", $("aqe-toggle").checked ? "true" : "false");
  body.append("geometric", $("geom-toggle").checked ? "true" : "false");
  try {
    state.results = await readJson(await fetch("/api/search", { method: "POST", body }));
    renderResults();
    renderMap();
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.searching = false;
    renderQuery();
  }
}

async function uploadFiles(fileList) {
  const files = [...fileList].filter((file) => file.type.startsWith("image/") || /\.(jpe?g|png|webp|gif|bmp|tiff?)$/i.test(file.name));
  if (!files.length) {
    toast("Those files are not images.", true);
    return;
  }
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  try {
    const result = await readJson(await fetch("/api/library", { method: "POST", body }));
    toast(`Added ${result.count} photograph${result.count === 1 ? "" : "s"}.`);
    await refreshLibrary();
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function deleteImage(path) {
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  try {
    await readJson(await fetch(`/api/library/${encoded}`, { method: "DELETE" }));
    await refreshLibrary();
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function changeModel(name) {
  const body = new FormData();
  body.append("model", name);
  try {
    await readJson(await fetch("/api/model", { method: "POST", body }));
    state.results = null;
    renderResults();
    await refreshStatus();
    await refreshSpace();
  } catch (error) {
    toast(error.message, true);
  }
}

async function doTextSearch() {
  const text = $("text-query").value.trim();
  if (!text || state.searching) return;
  state.searching = true;
  renderQuery();
  const body = new FormData();
  body.append("text", text);
  body.append("top_k", "12");
  body.append("aqe", $("aqe-toggle").checked ? "true" : "false");
  try {
    state.results = await readJson(await fetch("/api/search/text", { method: "POST", body }));
    renderResults();
    renderMap();
    setView("studio");
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.searching = false;
    renderQuery();
  }
}

async function doCompare() {
  if (!state.file || state.searching) return;
  state.searching = true;
  renderQuery();
  const body = new FormData();
  body.append("file", state.file);
  body.append("top_k", "6");
  try {
    state.compare = await readJson(await fetch("/api/compare", { method: "POST", body }));
    setView("compare");
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.searching = false;
    renderQuery();
  }
}

function renderCompare() {
  const grid = $("compare-grid");
  const payload = state.compare;
  if (!payload) {
    grid.innerHTML = "";
    const note = document.createElement("p");
    note.className = "fine";
    note.id = "compare-empty";
    note.textContent = "Choose a photograph in the studio, then use Compare models.";
    grid.appendChild(note);
    return;
  }
  $("compare-meta").textContent = "One query, three indexes";
  grid.replaceChildren();
  payload.columns.forEach((column) => {
    const card = document.createElement("article");
    card.className = "compare-col";
    const title = document.createElement("h2");
    title.textContent = column.model;
    card.appendChild(title);
    if (!column.indexed) {
      const note = document.createElement("p");
      note.className = "fine";
      note.textContent = "No index yet. Build this model from the library.";
      card.appendChild(note);
    } else {
      const list = document.createElement("ol");
      (column.results || []).forEach((result) => {
        const item = document.createElement("li");
        const img = document.createElement("img");
        img.src = result.url;
        img.alt = result.name;
        const caption = document.createElement("span");
        caption.textContent = `${result.similarity.toFixed(2)} · ${result.name}`;
        item.append(img, caption);
        list.appendChild(item);
      });
      card.appendChild(list);
    }
    grid.appendChild(card);
  });
}

async function loadEvaluation() {
  try {
    const data = await readJson(await fetch("/api/evaluate"));
    state.evalRows = data.results || [];
    renderEvaluation();
    if (data.state?.message && !data.state.running && data.state.error) {
      $("eval-status").textContent = data.state.error;
    }
  } catch (error) {
    $("eval-status").textContent = error.message;
  }
}

function metric(row, key) {
  const value = row[key];
  return typeof value === "number" ? value.toFixed(3) : "—";
}

function renderEvaluation() {
  const body = $("eval-body");
  body.replaceChildren();
  if (!state.evalRows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 10;
    cell.textContent = "No saved runs yet.";
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  state.evalRows.forEach((row) => {
    const tr = document.createElement("tr");
    [row.model, row.variant, metric(row, "precision@1"), metric(row, "precision@5"), metric(row, "precision@10"),
      metric(row, "recall@1"), metric(row, "recall@5"), metric(row, "recall@10"), metric(row, "mAP"), metric(row, "ndcg@10")]
      .forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.appendChild(cell);
      });
    body.appendChild(tr);
  });
}

async function startEvaluation() {
  const body = new FormData();
  body.append("model", $("eval-model").value);
  body.append("variant", $("eval-variant").value);
  try {
    await readJson(await fetch("/api/evaluate", { method: "POST", body }));
    state.wasEvaluating = true;
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function startBuild() {
  const body = new FormData();
  body.append("model", state.status?.active_model || "vgg19");
  try {
    await readJson(await fetch("/api/index/build", { method: "POST", body }));
    state.wasBuilding = true;
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

function bind() {
  document.querySelectorAll("[data-nav]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.nav));
  });
  $("status-pill").addEventListener("click", () => setView("library"));
  $("browse").addEventListener("click", () => $("file-input").click());
  $("replace").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", () => {
    const [file] = $("file-input").files;
    if (file) setFile(file);
    $("file-input").value = "";
  });
  $("search-btn").addEventListener("click", doSearch);
  $("compare-btn").addEventListener("click", doCompare);
  $("text-btn").addEventListener("click", doTextSearch);
  $("text-query").addEventListener("input", () => renderStatus(state.status));
  $("text-query").addEventListener("keydown", (event) => {
    if (event.key === "Enter") doTextSearch();
  });
  $("model-select").addEventListener("change", () => changeModel($("model-select").value));
  $("eval-btn").addEventListener("click", startEvaluation);
  $("k-range").addEventListener("input", () => {
    state.showCount = Number($("k-range").value);
    $("k-label").textContent = String(state.showCount);
    renderResults();
    renderMap();
  });

  const drop = $("drop");
  ["dragenter", "dragover"].forEach((name) => {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("is-over");
    });
  });
  drop.addEventListener("drop", (event) => {
    event.stopPropagation();
    const [file] = event.dataTransfer.files;
    if (file) setFile(file);
  });

  let dragDepth = 0;
  document.addEventListener("dragover", (event) => {
    if ([...event.dataTransfer.types].includes("Files")) event.preventDefault();
  });
  document.addEventListener("dragenter", (event) => {
    if (![...event.dataTransfer.types].includes("Files")) return;
    dragDepth += 1;
    if (state.view === "library") $("drop-overlay").hidden = false;
  });
  document.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) $("drop-overlay").hidden = true;
  });
  document.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    dragDepth = 0;
    $("drop-overlay").hidden = true;
    if (state.view !== "library" || !files?.length) return;
    event.preventDefault();
    uploadFiles(files);
  });

  window.addEventListener("paste", (event) => {
    const item = [...(event.clipboardData?.items || [])].find((entry) => entry.type.startsWith("image/"));
    if (!item || state.view !== "studio") return;
    const file = item.getAsFile();
    if (file) setFile(file);
  });

  $("add-btn").addEventListener("click", () => $("library-input").click());
  $("library-input").addEventListener("change", () => {
    uploadFiles($("library-input").files);
    $("library-input").value = "";
  });
  $("library-filter").addEventListener("input", () => {
    state.filter = $("library-filter").value;
    renderLibrary();
  });
  $("build-btn").addEventListener("click", startBuild);
  $("compare").addEventListener("click", (event) => {
    if (event.target === $("compare")) $("compare").close();
  });
  window.addEventListener("resize", () => {
    renderMap();
    if (state.results) drawFingerprint($("query-fp"), state.results.fingerprint);
  });
}

async function init() {
  bind();
  const hash = location.hash.replace("#", "");
  if (["library", "method", "studio", "compare", "evaluate"].includes(hash)) setView(hash);
  await Promise.all([refreshStatus(), refreshLibrary(), refreshSpace(), refreshSamples()]);
  setInterval(refreshStatus, 1200);
}

init();
