const app = document.querySelector("#app");
const state = {
  language: localStorage.getItem("ref2v-review-language") || "zh",
  filters: { search: "", status: "", sort: "score_asc", page: 1 },
  detail: null,
  frame: 0,
  searchTimer: null,
  listing: null,
  listingKey: "",
  listScrollY: 0,
  requestNumber: 0,
};

const messages = {
  zh: {
    product: "Ref2V 审核", cases: "评测案例", refresh: "刷新", search: "搜索 sample ID",
    status: "状态", all: "全部", complete: "已完成", error: "错误",
    low: "TextVScore 升序", high: "TextVScore 降序", extraSort: "额外文字比例降序",
    spatialSort: "空间保真度升序", temporalSort: "时间保真度升序",
    orderSort: "阅读顺序升序", idSort: "Sample ID", average: "平均 TextVScore",
    caseCount: "案例", errorCount: "错误", noCases: "没有匹配的案例", loading: "读取中...",
    failed: "读取失败", detail: "评测证据", back: "返回案例列表", score: "TextVScore",
    spatial: "空间保真度", temporal: "时间保真度", extraRatio: "额外文字比例",
    orderScore: "阅读顺序", references: "参考图与 Slots", frames: "视频与采样帧",
    noVideo: "视频不可用，仅显示采样帧", videoAvailable: "本地视频可播放",
    reference: "参考图", slot: "Slot", target: "目标文字", reading: "阅读顺序",
    targetScores: "目标评分", temporalEvidence: "逐目标时间证据", currentFrame: "当前帧",
    included: "参与聚合", excluded: "未参与聚合", activeRefs: "Active refs",
    noMatch: "Ref-ID 未匹配，当前帧不参与空间、时间和阅读顺序聚合",
    source: "原始帧", marker: "OCR Marker", batchMarker: "Matching Marker",
    candidates: "OCR Candidates", active: "Active", filtered: "已过滤", duplicateExtra: "重复 Extra",
    cappedExtra: "截断 Extra", extra: "Extra", confidence: "置信度", selected: "匹配到",
    predicted: "预测文字", matchScore: "分数", calls: "VLM 调用", route: "Ref-ID Routing",
    routeSkipped: "单参考图：跳过 Ref-ID Routing，直接使用唯一参考图",
    batch: "Target Matching", traceLoading: "读取 trace...", none: "无", rawData: "原始数据",
    config: "公开配置", run: "运行信息", time: "时间", noFrames: "没有采样帧",
    readingEvidence: "当前帧阅读顺序", referenceOrder: "Reference",
    predictedOrder: "Predicted", effectiveOrder: "缺失置末后的序列", missingSlots: "缺失 slots",
    wrongPairs: "错误 pairs", attempts: "全部尝试", payload: "User Payload", response: "最终响应",
    systemPrompt: "最终 System Prompt", retries: "VLM 重试", traceErrors: "最终调用错误",
    previous: "上一页", next: "下一页", page: "页", of: "共",
    zoomOut: "缩小", zoomIn: "放大", resetView: "重置视图", closeViewer: "关闭图片",
    evaluationError: "评测错误",
  },
  en: {
    product: "Ref2V Review", cases: "Evaluation Cases", refresh: "Refresh", search: "Search sample ID",
    status: "Status", all: "All", complete: "Complete", error: "Error",
    low: "TextVScore ascending", high: "TextVScore descending", extraSort: "Extra ratio descending",
    spatialSort: "Spatial fidelity ascending", temporalSort: "Temporal fidelity ascending",
    orderSort: "Reading order ascending", idSort: "Sample ID", average: "Average TextVScore",
    caseCount: "Cases", errorCount: "Errors", noCases: "No matching cases", loading: "Loading...",
    failed: "Failed to load", detail: "Evaluation Evidence", back: "Back to cases", score: "TextVScore",
    spatial: "Spatial fidelity", temporal: "Temporal fidelity", extraRatio: "Extra text ratio",
    orderScore: "Reading order", references: "References and Slots", frames: "Video and Sampled Frames",
    noVideo: "Video unavailable; showing sampled frames", videoAvailable: "Local video available",
    reference: "Reference", slot: "Slot", target: "Target text", reading: "Reading order",
    targetScores: "Target Scores", temporalEvidence: "Per-target Temporal Evidence", currentFrame: "Current Frame",
    included: "Included", excluded: "Excluded", activeRefs: "Active refs",
    noMatch: "No active reference; excluded from spatial, temporal, and reading-order aggregation",
    source: "Source frame", marker: "OCR Marker", batchMarker: "Matching Marker",
    candidates: "OCR Candidates", active: "Active", filtered: "Filtered", duplicateExtra: "Duplicate extra",
    cappedExtra: "Capped extra", extra: "Extra", confidence: "Confidence", selected: "Matched",
    predicted: "Predicted text", matchScore: "Score", calls: "VLM Calls", route: "Ref-ID Routing",
    routeSkipped: "Single reference: Ref-ID routing was skipped and the only reference was used directly",
    batch: "Target Matching", traceLoading: "Loading trace...", none: "None", rawData: "Raw Data",
    config: "Public Config", run: "Run Details", time: "Time", noFrames: "No sampled frames",
    readingEvidence: "Current-frame Reading Order", referenceOrder: "Reference",
    predictedOrder: "Predicted", effectiveOrder: "Effective (missing last)", missingSlots: "Missing slots",
    wrongPairs: "Incorrect pairs", attempts: "All Attempts", payload: "User Payload", response: "Final Response",
    systemPrompt: "Final System Prompt", retries: "VLM retries", traceErrors: "Final call errors",
    previous: "Previous", next: "Next", page: "Page", of: "of",
    zoomOut: "Zoom out", zoomIn: "Zoom in", resetView: "Reset view", closeViewer: "Close image",
    evaluationError: "Evaluation error",
  },
};

const t = (key) => messages[state.language][key] || key;
const escapeHtml = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
);
const number = (value, digits = 2) => value == null || Number.isNaN(Number(value)) ? "-" : Number(value).toFixed(digits);
const percent = (value) => value == null ? "-" : `${(Number(value) * 100).toFixed(1)}%`;
const pretty = (value) => JSON.stringify(value ?? {}, null, 2);
const apiUrl = (path) => `api${path}`;
const mediaUrl = (path) => `media${path}`;

async function getJson(path) {
  const response = await fetch(path);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}

function metric(label, value, className = "") {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong class="${className}">${escapeHtml(value)}</strong></div>`;
}

function badge(status) {
  const labels = { complete: t("complete"), error: t("error") };
  return `<span class="badge ${escapeHtml(status)}">${escapeHtml(labels[status] || status)}</span>`;
}

function casePath(sampleId) {
  return `/cases/${encodeURIComponent(sampleId)}`;
}

function listQuery() {
  return new URLSearchParams({ ...state.filters, page_size: 50 });
}

function renderListing(listing) {
  const listBody = document.querySelector("#list-body");
  if (!listBody) return;
  const counts = listing.counts || {};
  const rows = listing.items.map((item) => `
      <tr tabindex="0" data-sample="${escapeHtml(item.sample_id)}">
        <td class="mono">${escapeHtml(item.sample_id)}</td>
        <td class="score ${Number(item.metrics?.textv_score) < 60 ? "bad" : ""}">${number(item.metrics?.textv_score, 1)}</td>
        <td>${percent(item.metrics?.length_weighted_spatial_fidelity)}</td>
        <td>${percent(item.metrics?.length_weighted_temporal_fidelity)}</td>
        <td>${percent(item.metrics?.extra_text_ratio)}</td>
        <td>${percent(item.metrics?.reading_order_score)}</td>
        <td>${item.reference_count}</td><td>${item.frame_count}</td><td>${badge(item.status)}</td>
      </tr>`).join("");
  listBody.innerHTML = `
      <div class="metrics">
        ${metric(t("average"), number(counts.average_score, 1))}
        ${metric(t("caseCount"), counts.cases || 0)}
        ${metric(t("errorCount"), counts.error || 0, counts.error ? "bad" : "")}
      </div>
      <div class="toolbar">
        <input id="search" value="${escapeHtml(state.filters.search)}" placeholder="${escapeHtml(t("search"))}" aria-label="${escapeHtml(t("search"))}">
        <select id="status"><option value="">${escapeHtml(t("all"))}</option><option value="complete">${escapeHtml(t("complete"))}</option><option value="error">${escapeHtml(t("error"))}</option></select>
        <select id="sort"><option value="score_asc">${escapeHtml(t("low"))}</option><option value="score_desc">${escapeHtml(t("high"))}</option><option value="spatial_asc">${escapeHtml(t("spatialSort"))}</option><option value="temporal_asc">${escapeHtml(t("temporalSort"))}</option><option value="extra_desc">${escapeHtml(t("extraSort"))}</option><option value="reading_asc">${escapeHtml(t("orderSort"))}</option><option value="id">${escapeHtml(t("idSort"))}</option></select>
      </div>
      ${rows ? `<div class="table-wrap"><table><thead><tr><th>sample_id</th><th>${escapeHtml(t("score"))}</th><th>${escapeHtml(t("spatial"))}</th><th>${escapeHtml(t("temporal"))}</th><th>${escapeHtml(t("extraRatio"))}</th><th>${escapeHtml(t("orderScore"))}</th><th>Refs</th><th>Frames</th><th>${escapeHtml(t("status"))}</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="pagination"><span>${escapeHtml(t("page"))} ${listing.page} ${escapeHtml(t("of"))} ${Math.max(1, Math.ceil(listing.total / listing.page_size))} · ${listing.total}</span><div><button id="previous-page" class="command" type="button" ${listing.page <= 1 ? "disabled" : ""}>← ${escapeHtml(t("previous"))}</button><button id="next-page" class="command" type="button" ${listing.page * listing.page_size >= listing.total ? "disabled" : ""}>${escapeHtml(t("next"))} →</button></div></div>` : `<div class="empty">${escapeHtml(t("noCases"))}</div>`}`;

  const search = document.querySelector("#search");
  search.addEventListener("input", () => {
    state.filters.search = search.value;
    state.filters.page = 1;
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(loadList, 250);
  });
  document.querySelector("#status").value = state.filters.status;
  document.querySelector("#status").addEventListener("change", (event) => { state.filters.status = event.target.value; state.filters.page = 1; loadList(); });
  document.querySelector("#sort").value = state.filters.sort;
  document.querySelector("#sort").addEventListener("change", (event) => { state.filters.sort = event.target.value; state.filters.page = 1; loadList(); });
  document.querySelector("#previous-page")?.addEventListener("click", () => { state.filters.page -= 1; loadList(); });
  document.querySelector("#next-page")?.addEventListener("click", () => { state.filters.page += 1; loadList(); });
  document.querySelectorAll("tr[data-sample]").forEach((row) => {
    const open = () => openCase(row.dataset.sample);
    row.addEventListener("click", open);
    row.addEventListener("keydown", (event) => { if (event.key === "Enter") open(); });
  });
}

function renderList() {
  state.detail = null;
  ++state.requestNumber;
  app.innerHTML = `
    <div class="page-heading">
      <div><div class="overline">${escapeHtml(t("product").toUpperCase())}</div><h1>${escapeHtml(t("cases"))}</h1></div>
      <button id="refresh" class="command" type="button" title="${escapeHtml(t("refresh"))}" aria-label="${escapeHtml(t("refresh"))}">↻</button>
    </div>
    <div id="list-body"><div class="empty">${escapeHtml(t("loading"))}</div></div>`;
  document.querySelector("#refresh").addEventListener("click", () => loadList(true));
  if (state.listing && state.listingKey === listQuery().toString()) {
    renderListing(state.listing);
    requestAnimationFrame(() => window.scrollTo(0, state.listScrollY));
  } else {
    loadList();
  }
}

async function loadList(forceRefresh = false) {
  const request = ++state.requestNumber;
  const query = listQuery();
  const key = query.toString();
  if (forceRefresh) query.set("refresh", "1");
  try {
    const listing = await getJson(`${apiUrl("/cases")}?${query}`);
    if (request !== state.requestNumber || !document.querySelector("#list-body")) return;
    state.listing = listing;
    state.listingKey = key;
    renderListing(listing);
  } catch (error) {
    const listBody = document.querySelector("#list-body");
    if (request === state.requestNumber && listBody) listBody.innerHTML = `<div class="notice error">${escapeHtml(t("failed"))}: ${escapeHtml(error.message)}</div>`;
  }
}

function openCase(sampleId) {
  state.listScrollY = window.scrollY;
  history.replaceState({ ...(history.state || {}), view: "list" }, "", location.href);
  history.pushState({ view: "case", fromList: true }, "", `?case=${encodeURIComponent(sampleId)}`);
  loadCase(sampleId);
}

function backToList() {
  if (history.state?.fromList) {
    history.back();
  } else {
    history.replaceState({ view: "list" }, "", location.pathname);
    renderList();
  }
}

async function loadCase(sampleId) {
  const request = ++state.requestNumber;
  app.innerHTML = `<div class="empty">${escapeHtml(t("loading"))}</div>`;
  try {
    const detail = await getJson(apiUrl(casePath(sampleId)));
    if (request !== state.requestNumber) return;
    state.detail = detail;
    state.frame = 0;
    renderCase();
  } catch (error) {
    if (request === state.requestNumber) app.innerHTML = `<div class="notice error">${escapeHtml(t("failed"))}: ${escapeHtml(error.message)}</div>`;
  }
}

function renderCase() {
  const detail = state.detail;
  const summary = detail.summary;
  const metrics = summary.metrics || {};
  const frames = detail.frames || [];
  app.innerHTML = `
    <button id="back" class="text-command" type="button">← ${escapeHtml(t("back"))}</button>
    <div class="page-heading"><div><div class="overline">${escapeHtml(t("detail").toUpperCase())}</div><h1>${escapeHtml(summary.sample_id)}</h1>
      <div class="detail-meta"><span>${number(summary.duration_sec, 3)} s</span><span>${frames.length} frames</span><span>${summary.reference_count} references</span><span>${summary.evaluated_frames} ${escapeHtml(t("included"))}</span><span>${summary.no_match_frames} ${escapeHtml(t("excluded"))}</span><span>${summary.vlm_retries} ${escapeHtml(t("retries"))}</span><span>${summary.trace_errors} ${escapeHtml(t("traceErrors"))}</span></div></div>${badge(summary.status)}</div>
    ${summary.error ? `<div class="notice error"><strong>${escapeHtml(t("evaluationError"))}</strong><br>${escapeHtml(summary.error.error || pretty(summary.error))}</div>` : ""}
    <div class="metrics">${metric(t("score"), number(metrics.textv_score, 1), Number(metrics.textv_score) < 60 ? "bad" : "")}${metric(t("spatial"), percent(metrics.length_weighted_spatial_fidelity))}${metric(t("temporal"), percent(metrics.length_weighted_temporal_fidelity))}${metric(t("extraRatio"), percent(metrics.extra_text_ratio), Number(metrics.extra_text_ratio) > 0.25 ? "bad" : "")}${metric(t("orderScore"), percent(metrics.reading_order_score))}</div>
    <div class="detail-grid">
      <div>
        <section class="section"><h2>${escapeHtml(t("references"))}</h2>${renderReferences(detail.references || [], metrics)}</section>
        <section class="section"><div class="section-head"><h2>${escapeHtml(t("frames"))}</h2><span class="subtle">${escapeHtml(summary.video_available ? t("videoAvailable") : t("noVideo"))}</span></div>
          ${summary.video_available ? `<video class="video" controls preload="metadata" src="${mediaUrl(`${casePath(summary.sample_id)}/video`)}"></video>` : ""}
          <div class="frames">${frames.map((frame, index) => `<button class="frame-button ${index === state.frame ? "active" : ""}" data-frame="${index}" type="button"><span>${number(frame.timestamp, 3)} s</span><small>#${frame.frame_index ?? "-"}</small></button>`).join("")}</div><div id="frame-detail"></div>
        </section>
      </div>
      <div>
        <section class="section"><h2>${escapeHtml(t("targetScores"))}</h2>${renderTargets(metrics.targets || [])}</section>
        <section class="section"><h2>${escapeHtml(t("temporalEvidence"))}</h2>${renderTemporal(metrics.targets || [], frames)}</section>
        <section class="section"><h2>${escapeHtml(t("rawData"))}</h2><select id="raw-select"><option value="result">result</option><option value="manifest">manifest</option><option value="input_record">input_record</option><option value="config">${escapeHtml(t("config"))}</option><option value="run_details">${escapeHtml(t("run"))}</option><option value="frame">${escapeHtml(t("currentFrame"))}</option></select><pre id="raw-json"></pre></section>
      </div>
    </div>`;

  document.querySelector("#back").addEventListener("click", backToList);
  document.querySelectorAll("[data-frame]").forEach((button) => button.addEventListener("click", () => {
    state.frame = Number(button.dataset.frame);
    document.querySelectorAll("[data-frame]").forEach((item) => item.classList.toggle("active", item === button));
    renderFrame();
    renderRaw();
    bindImages(document.querySelector("#frame-detail"));
  }));
  document.querySelector("#raw-select").addEventListener("change", renderRaw);
  renderRaw();
  if (frames.length) renderFrame();
  else document.querySelector("#frame-detail").innerHTML = `<div class="empty">${escapeHtml(t("noFrames"))}</div>`;
  bindImages();
}

function renderReferences(references, metrics) {
  if (!references.length) return `<div class="empty">${escapeHtml(t("none"))}</div>`;
  const sampleId = state.detail.summary.sample_id;
  return references.map((reference) => {
    const refMetric = (metrics.refs || []).find((item) => item.ref_id === reference.ref_id) || {};
    const rows = (reference.slots || []).map((slot) => {
      const target = (metrics.targets || []).find((item) => item.slot_id === slot.slot_id) || {};
      return `<tr><td class="mono">${escapeHtml(slot.slot_id)}</td><td>${escapeHtml(slot.target_text)}</td><td>${slot.reading_order ?? "-"}</td><td>${percent(target.spatial)}</td><td>${percent(target.temporal)}</td></tr>`;
    }).join("");
    const src = reference.image ? mediaUrl(`${casePath(sampleId)}/references/${encodeURIComponent(reference.image)}`) : null;
    return `<article class="reference"><div class="reference-head"><h3>${escapeHtml(reference.ref_id)}</h3><span class="subtle">${escapeHtml(reference.filename || "")}</span></div><div class="reference-metrics"><span>${escapeHtml(t("spatial"))}: <strong>${percent(refMetric.spatial)}</strong></span><span>${escapeHtml(t("temporal"))}: <strong>${percent(refMetric.temporal)}</strong></span><span>${escapeHtml(t("orderScore"))}: <strong>${percent(refMetric.reading_order)}</strong></span></div>${src ? `<button class="image-open" data-image="${src}" data-title="${escapeHtml(`${t("reference")} ${reference.ref_id}`)}"><img src="${src}" alt="${escapeHtml(reference.ref_id)}"></button>` : ""}<div class="table-scroll"><table class="mini"><thead><tr><th>${escapeHtml(t("slot"))}</th><th>${escapeHtml(t("target"))}</th><th>${escapeHtml(t("reading"))}</th><th>${escapeHtml(t("spatial"))}</th><th>${escapeHtml(t("temporal"))}</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
  }).join("");
}

function renderTargets(targets) {
  if (!targets.length) return `<div class="empty">${escapeHtml(t("none"))}</div>`;
  return `<div class="table-scroll"><table class="mini"><thead><tr><th>${escapeHtml(t("slot"))}</th><th>${escapeHtml(t("target"))}</th><th>ref_id</th><th>${escapeHtml(t("reading"))}</th><th>${escapeHtml(t("spatial"))}</th><th>${escapeHtml(t("temporal"))}</th></tr></thead><tbody>${targets.map((item) => `<tr><td class="mono">${escapeHtml(item.slot_id)}</td><td>${escapeHtml(item.target_text)}</td><td>${escapeHtml(item.ref_id)}</td><td>${item.reading_order ?? "-"}</td><td>${percent(item.spatial)}</td><td>${percent(item.temporal)}</td></tr>`).join("")}</tbody></table></div>`;
}

function renderTemporal(targets, frames) {
  if (!targets.length) return `<div class="empty">${escapeHtml(t("none"))}</div>`;
  return targets.map((target) => {
    const rows = frames.filter((frame) => frame.evaluated).map((frame) => {
      const match = (frame.matches || []).find((item) => item.slot_id === target.slot_id);
      return `<tr><td>${number(frame.timestamp, 3)} s</td><td>${escapeHtml(match?.pred_text || "-")}</td><td>${match?.score == null ? "-" : number(match.score, 3)}</td><td>${escapeHtml((match?.selected_ids || []).join(", ") || "-")}</td></tr>`;
    }).join("");
    return `<details class="trace"><summary>${escapeHtml(target.slot_id)} · ${escapeHtml(target.target_text)}</summary><div class="trace-body"><div class="table-scroll"><table class="mini"><thead><tr><th>${escapeHtml(t("time"))}</th><th>${escapeHtml(t("predicted"))}</th><th>${escapeHtml(t("matchScore"))}</th><th>ID</th></tr></thead><tbody>${rows}</tbody></table></div></div></details>`;
  }).join("");
}

function renderRaw() {
  const detail = state.detail;
  const values = {
    result: detail.result, manifest: detail.manifest, input_record: detail.input_record,
    config: detail.config, run_details: detail.run_details, frame: detail.frames[state.frame] || {},
  };
  document.querySelector("#raw-json").textContent = pretty(values[document.querySelector("#raw-select").value]);
}

function renderFrame() {
  const detail = state.detail;
  const frame = detail.frames[state.frame];
  const sampleId = detail.summary.sample_id;
  const files = new Set(frame.files || []);
  const image = (name, label) => files.has(name) ? `<figure><button class="image-open" data-image="${mediaUrl(`${casePath(sampleId)}/frames/${encodeURIComponent(frame.folder)}/${encodeURIComponent(name)}`)}" data-title="${escapeHtml(label)}"><img src="${mediaUrl(`${casePath(sampleId)}/frames/${encodeURIComponent(frame.folder)}/${encodeURIComponent(name)}`)}" alt="${escapeHtml(label)}"></button><figcaption>${escapeHtml(label)}</figcaption></figure>` : "";

  const selected = new Map();
  (frame.matches || []).forEach((match) => (match.selected_ids || []).forEach((id) => {
    if (!selected.has(id)) selected.set(id, []);
    selected.get(id).push(match.slot_id);
  }));
  const active = new Set(frame.filtering?.active_ids || []);
  const duplicate = new Set(frame.filtering?.duplicate_extra_ids || []);
  const capped = new Set(frame.filtering?.capped_extra_ids || []);
  const candidateRows = (frame.ocr_candidates || []).map((candidate) => {
    const targets = selected.get(candidate.id) || [];
    const isExtra = (frame.extra_ids || []).includes(candidate.id);
    const status = duplicate.has(candidate.id) ? t("duplicateExtra") : capped.has(candidate.id) ? t("cappedExtra") : active.has(candidate.id) ? t("active") : t("filtered");
    return `<tr><td class="mono">${escapeHtml(candidate.id)}</td><td>${escapeHtml(candidate.ocr_text)}</td><td>${number(candidate.confidence, 3)}</td><td>${escapeHtml(status)}</td><td class="${targets.length ? "good" : isExtra ? "bad" : ""}">${targets.length ? escapeHtml(targets.join(", ")) : isExtra ? escapeHtml(t("extra")) : "-"}</td></tr>`;
  }).join("");
  const matchRows = (frame.matches || []).map((match) => `<tr><td class="mono">${escapeHtml(match.slot_id)}</td><td>${escapeHtml(match.ref_id || "-")}</td><td>${escapeHtml((match.selected_ids || []).join(", ") || "-")}</td><td>${escapeHtml(match.pred_text || "-")}</td><td>${match.score == null ? "-" : number(match.score, 3)}</td></tr>`).join("");
  const reading = (frame.reading_order || []).map((item) => `<div class="reading"><strong>${escapeHtml(item.ref_id)} · ${percent(item.score)}</strong><div><span>${escapeHtml(t("referenceOrder"))}</span><code>${escapeHtml(item.reference.join(" → ") || "-")}</code></div><div><span>${escapeHtml(t("predictedOrder"))}</span><code>${escapeHtml(item.predicted.join(" → ") || "-")}</code></div><div><span>${escapeHtml(t("effectiveOrder"))}</span><code>${escapeHtml(item.effective.join(" → ") || "-")}</code></div><div><span>${escapeHtml(t("missingSlots"))}</span><code>${escapeHtml(item.missing.join(", ") || "-")}</code></div><div><span>${escapeHtml(t("wrongPairs"))}</span><code>${escapeHtml(item.incorrect_pairs.map((pair) => `${pair.left}>${pair.right}`).join(", ") || "-")}</code></div></div>`).join("");
  const traceFiles = (frame.files || []).filter((name) => /(?:ref_id_matching_vlm_call|matching_vlm_call_batch_\d+)\.json$/.test(name));
  const routeSkipped = detail.references.length === 1 && !traceFiles.some((name) => name.startsWith("ref_id"));
  const matchingImages = [...files].filter((name) => /^matching_marker_batch_\d+\.jpg$/.test(name));

  document.querySelector("#frame-detail").innerHTML = `
    <div class="section-head"><h3>${escapeHtml(t("currentFrame"))}: ${number(frame.timestamp, 3)} s · #${frame.frame_index ?? "-"}</h3><span class="${frame.evaluated ? "good" : "bad"}">${escapeHtml(frame.evaluated ? t("included") : t("excluded"))}</span></div>
    ${frame.evaluated ? "" : `<div class="notice">${escapeHtml(t("noMatch"))}</div>`}
    <div class="frame-visuals">${image("source_frame.jpg", t("source"))}${image("marker.jpg", t("marker"))}${matchingImages.map((name) => image(name, `${t("batchMarker")} ${name.match(/\d+/)?.[0] || ""}`)).join("")}</div>
    <div class="evidence-grid">
      <section><h3>${escapeHtml(t("candidates"))}</h3><div class="table-scroll"><table class="mini"><thead><tr><th>ID</th><th>OCR</th><th>${escapeHtml(t("confidence"))}</th><th>${escapeHtml(t("status"))}</th><th>${escapeHtml(t("selected"))}</th></tr></thead><tbody>${candidateRows || `<tr><td colspan="5">${escapeHtml(t("none"))}</td></tr>`}</tbody></table></div><p class="evidence-note">Extra ratio: ${percent(frame.extra_text_ratio)} · ${escapeHtml(t("activeRefs"))}: ${escapeHtml((frame.matched_ref_ids || []).join(", ") || "-")}</p></section>
      <section><h3>${escapeHtml(t("batch"))}</h3><div class="table-scroll"><table class="mini"><thead><tr><th>${escapeHtml(t("slot"))}</th><th>ref_id</th><th>ID</th><th>${escapeHtml(t("predicted"))}</th><th>${escapeHtml(t("matchScore"))}</th></tr></thead><tbody>${matchRows || `<tr><td colspan="5">${escapeHtml(t("none"))}</td></tr>`}</tbody></table></div></section>
    </div>
    ${reading ? `<section><h3>${escapeHtml(t("readingEvidence"))}</h3>${reading}</section>` : ""}
    <section><h3>${escapeHtml(t("calls"))}</h3>${routeSkipped ? `<p class="evidence-note">${escapeHtml(t("routeSkipped"))}</p>` : ""}${traceFiles.length ? traceFiles.map((name) => `<details class="trace" data-trace="${escapeHtml(name)}"><summary>${escapeHtml(name.includes("ref_id") ? t("route") : `${t("batch")} ${name.match(/\d+/)?.[0] || ""}`)}</summary><div class="trace-body">${escapeHtml(t("traceLoading"))}</div></details>`).join("") : `<p class="subtle">${escapeHtml(t("none"))}</p>`}</section>`;

  document.querySelectorAll(".trace[data-trace]").forEach((node) => node.addEventListener("toggle", async () => {
    if (!node.open || node.dataset.loaded) return;
    const body = node.querySelector(".trace-body");
    try {
      const trace = await getJson(apiUrl(`${casePath(sampleId)}/frames/${encodeURIComponent(frame.folder)}/${encodeURIComponent(node.dataset.trace)}`));
      body.innerHTML = `${trace.error ? `<p class="bad">${escapeHtml(trace.error)}</p>` : ""}<div class="trace-block"><label>${escapeHtml(t("systemPrompt"))}</label><pre>${escapeHtml(trace.final_prompt || "-")}</pre></div><div class="trace-block"><label>${escapeHtml(t("payload"))}</label><pre>${escapeHtml(pretty(trace.payload))}</pre></div><div class="trace-block"><label>${escapeHtml(t("response"))}</label><pre>${escapeHtml(pretty(trace.final_response))}</pre></div><div class="trace-block"><label>${escapeHtml(t("attempts"))}</label><pre>${escapeHtml(pretty(trace.attempts))}</pre></div>`;
      node.dataset.loaded = "true";
    } catch (error) {
      body.textContent = `${t("failed")}: ${error.message}`;
    }
  }));
}

function bindImages(root = document) {
  root.querySelectorAll(".image-open").forEach((button) => button.addEventListener("click", () => openViewer(button.dataset.image, button.dataset.title)));
}

const viewer = document.querySelector("#image-viewer");
const viewerImage = document.querySelector("#viewer-image");
const viewerViewport = document.querySelector("#viewer-viewport");
const MIN_VIEWER_SCALE = 0.1;
let viewerScale = 1;
let viewerOffset = { x: 0, y: 0 };
let viewerDrag = null;
function updateViewer() {
  viewerImage.style.transform = `translate(${viewerOffset.x}px, ${viewerOffset.y}px) scale(${viewerScale})`;
  viewerViewport.classList.toggle("can-pan", viewerScale > 1);
  document.querySelector("#viewer-scale").textContent = `${Math.round(viewerScale * 100)}%`;
}
function resetViewer() { viewerScale = 1; viewerOffset = { x: 0, y: 0 }; updateViewer(); }
function openViewer(src, title) { viewerImage.src = src; viewerImage.alt = title; document.querySelector("#viewer-title").textContent = title; resetViewer(); viewer.showModal(); }
document.querySelector("#viewer-in").addEventListener("click", () => { viewerScale = Math.min(8, viewerScale * 1.35); updateViewer(); });
document.querySelector("#viewer-out").addEventListener("click", () => { viewerScale = Math.max(MIN_VIEWER_SCALE, viewerScale / 1.35); if (viewerScale <= 1) viewerOffset = { x: 0, y: 0 }; updateViewer(); });
document.querySelector("#viewer-reset").addEventListener("click", resetViewer);
document.querySelector("#viewer-close").addEventListener("click", () => viewer.close());
viewerViewport.addEventListener("wheel", (event) => { event.preventDefault(); viewerScale = Math.min(8, Math.max(MIN_VIEWER_SCALE, viewerScale * (event.deltaY < 0 ? 1.15 : 0.87))); if (viewerScale <= 1) viewerOffset = { x: 0, y: 0 }; updateViewer(); }, { passive: false });
viewerViewport.addEventListener("dblclick", () => { viewerScale = viewerScale > 1 ? 1 : 2; viewerOffset = { x: 0, y: 0 }; updateViewer(); });
viewerViewport.addEventListener("pointerdown", (event) => {
  if (viewerScale <= 1) return;
  viewerDrag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, originX: viewerOffset.x, originY: viewerOffset.y };
  viewerViewport.setPointerCapture(event.pointerId);
  viewerViewport.classList.add("panning");
});
viewerViewport.addEventListener("pointermove", (event) => {
  if (!viewerDrag || viewerDrag.pointerId !== event.pointerId) return;
  viewerOffset = {
    x: viewerDrag.originX + event.clientX - viewerDrag.startX,
    y: viewerDrag.originY + event.clientY - viewerDrag.startY,
  };
  updateViewer();
});
viewerViewport.addEventListener("pointerup", () => { viewerDrag = null; viewerViewport.classList.remove("panning"); });

function applyLanguage() {
  document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
  document.querySelector("#product-label").textContent = t("product");
  document.querySelectorAll("#language-switch button").forEach((button) => button.classList.toggle("active", button.dataset.language === state.language));
  [["#viewer-out", "zoomOut"], ["#viewer-in", "zoomIn"], ["#viewer-reset", "resetView"], ["#viewer-close", "closeViewer"]].forEach(([selector, key]) => {
    const button = document.querySelector(selector);
    button.title = t(key);
    button.setAttribute("aria-label", t(key));
  });
}
document.querySelectorAll("#language-switch button").forEach((button) => button.addEventListener("click", () => {
  state.language = button.dataset.language;
  localStorage.setItem("ref2v-review-language", state.language);
  applyLanguage();
  route();
}));

function route() {
  const sampleId = new URLSearchParams(location.search).get("case");
  if (sampleId && /^[A-Za-z0-9_.-]+$/.test(sampleId)) loadCase(sampleId);
  else renderList();
}
window.addEventListener("popstate", route);
applyLanguage();
route();
