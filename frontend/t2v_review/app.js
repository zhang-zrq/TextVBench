const app = document.querySelector("#app");
const filters = { search: "", benchmark: "", language: "", status: "", sort: "score_asc", page: 1 };
let listing = null;
let listingKey = "";
let detail = null;
let selectedFrame = 0;
let requestNumber = 0;
let searchTimer;
let listScrollY = 0;

const messages = {
  zh: {
    documentTitle: "TextVBench · Evaluator 审核", productName: "Evaluator 审核", headerLabel: "评测证据",
    languageControl: "界面语言", reviewBundles: "审核数据包", evaluationCases: "评测案例", refresh: "刷新", caseList: "案例列表",
    searchCases: "搜索案例", searchPlaceholder: "搜索 sample ID / pair ID", benchmark: "子 benchmark", allBenchmarks: "全部 benchmark",
    language: "语言", allLanguages: "全部语言", status: "状态", allStatuses: "全部状态", statusComplete: "已完成", statusError: "调用异常",
    sort: "排序", scoreAscending: "TextVScore 升序", scoreDescending: "TextVScore 降序", extraDescending: "额外文字比例降序",
    sampleIdSort: "Sample ID", loadingCases: "读取案例中...", cases: "案例", averageScore: "平均 TextVScore", belowSixty: "低于 60 分",
    evaluationErrors: "评测错误", matchedCount: "匹配 {count} 条", extraText: "额外文字", sampledFrames: "采样帧", sourceQc: "源数据 QC",
    viewCase: "查看 {id}", noCases: "没有匹配的案例", pageCount: "第 {page} 页 / 共 {total} 页", previous: "上一页", next: "下一页",
    readFailed: "读取失败：{error}", sampleId: "样本 ID", textVScore: "TextVScore", spatial: "空间", temporal: "时间",
    spatialFidelity: "空间保真度", temporalFidelity: "时间保真度", extraTextRatio: "额外文字比例", manifest: "评测清单",
    carrier: "载体", slot: "文本槽位", target: "目标", textCarrier: "文字 / 载体", noTargets: "无目标信息", loadingEvidence: "读取评测证据中...",
    backToCases: "返回案例列表", framesCount: "{count} 帧", videoAndFrames: "视频与采样帧", videoAvailable: "本地视频可播放",
    videoUnavailable: "无视频文件，查看采样帧", viewFrame: "查看 {time} 秒的采样帧", rawData: "原始数据", rawDataType: "原始数据类型",
    inputRecord: "输入记录", fullResult: "完整结果", publicConfig: "公开配置摘要", currentFrameResult: "当前帧结果", targetsAndScores: "目标与评分",
    currentFrameDecision: "当前帧判断", vlmCalls: "VLM 调用", noFrames: "没有采样帧", timeAndFrame: "时间 {time} s · 帧 #{frame}",
    sourceFrame: "原始帧", sourceFramePurpose: "未经标注的采样视频帧", carrierMarker: "载体标记图 · VLM 参考图",
    carrierMarkerPurpose: "阶段 1：展示当前帧全部 active OCR 候选，判断载体是否可见并完成候选归属", matchingMarker: "匹配标记图 · VLM 参考图",
    matchingMarkerPurpose: "阶段 2 · 批次 {batch}：仅展示载体归属后尚未被快速匹配消费的候选，并匹配到目标文字", openImage: "放大查看 {title}",
    visible: "可见", notVisible: "不可见", noRelatedCandidates: "无关联候选", noCarrierResults: "无载体结果", targetMatching: "目标匹配",
    predictedText: "预测文字", candidateId: "候选 ID", score: "分数", carrierMissing: "N/A · 载体缺失", noMatchResults: "无匹配结果",
    ocrCandidates: "OCR 候选", recognizedText: "识别文字", confidence: "置信度", candidateStatus: "状态", candidateExtra: "额外文字",
    candidateActive: "参与匹配", candidateFiltered: "已过滤", noOcrCandidates: "本帧无 OCR 候选", frameExtraRatio: "本帧额外文字比例 {ratio} · Extra IDs: {ids}",
    none: "无", carrierDecision: "载体判断", targetBatch: "目标匹配 · 批次 {batch}", loadingCall: "读取调用记录中...",
    noVlmCalls: "没有 VLM 调用记录", fastMatched: "本帧目标由快速匹配解决，或没有可供 VLM 选择的候选。", finalSystemPrompt: "最终 System Prompt",
    userPayload: "User Payload", finalResponse: "最终响应", allAttempts: "全部尝试 ({count})", zoomOut: "缩小", zoomIn: "放大",
    resetView: "复位视图", closeViewer: "关闭图片查看器", imageViewerHint: "滚轮或 +/- 缩放 · 拖拽移动 · 双击切换放大",
  },
  en: {
    documentTitle: "TextVBench · Evaluator Review", productName: "Evaluator Review", headerLabel: "Evaluation Evidence",
    languageControl: "Interface language", reviewBundles: "REVIEW BUNDLES", evaluationCases: "Evaluation Cases", refresh: "Refresh", caseList: "Case List",
    searchCases: "Search cases", searchPlaceholder: "Search sample ID / pair ID", benchmark: "Sub-benchmark", allBenchmarks: "All benchmarks",
    language: "Language", allLanguages: "All languages", status: "Status", allStatuses: "All statuses", statusComplete: "Complete", statusError: "Call error",
    sort: "Sort", scoreAscending: "TextVScore ascending", scoreDescending: "TextVScore descending", extraDescending: "Extra text descending",
    sampleIdSort: "Sample ID", loadingCases: "Loading cases...", cases: "Cases", averageScore: "Average TextVScore", belowSixty: "Below 60",
    evaluationErrors: "Evaluation errors", matchedCount: "{count} matches", extraText: "Extra text", sampledFrames: "Frames", sourceQc: "Source QC",
    viewCase: "View {id}", noCases: "No matching cases", pageCount: "Page {page} of {total}", previous: "Previous", next: "Next",
    readFailed: "Failed to load: {error}", sampleId: "Sample ID", textVScore: "TextVScore", spatial: "Spatial", temporal: "Temporal",
    spatialFidelity: "Spatial Fidelity", temporalFidelity: "Temporal Fidelity", extraTextRatio: "Extra Text Ratio", manifest: "Manifest",
    carrier: "Carrier", slot: "Slot", target: "Target", textCarrier: "Text / carrier", noTargets: "No target information", loadingEvidence: "Loading evaluation evidence...",
    backToCases: "Back to case list", framesCount: "{count} frames", videoAndFrames: "Video and Sampled Frames", videoAvailable: "Local video available",
    videoUnavailable: "Video unavailable; showing sampled frames", viewFrame: "View sampled frame at {time} seconds", rawData: "Raw Data", rawDataType: "Raw data type",
    inputRecord: "Input record", fullResult: "Full result", publicConfig: "Public config summary", currentFrameResult: "Current frame result", targetsAndScores: "Targets and Scores",
    currentFrameDecision: "Current Frame Decision", vlmCalls: "VLM Calls", noFrames: "No sampled frames", timeAndFrame: "Time {time} s · frame #{frame}",
    sourceFrame: "Source frame", sourceFramePurpose: "Unannotated sampled video frame", carrierMarker: "Carrier marker · VLM reference",
    carrierMarkerPurpose: "Stage 1: show all active OCR candidates, verify carrier visibility, and assign candidates to carriers", matchingMarker: "Matching marker · VLM reference",
    matchingMarkerPurpose: "Stage 2 · Batch {batch}: show only carrier-associated candidates not consumed by fast matching, then match them to targets", openImage: "Open enlarged view of {title}",
    visible: "Visible", notVisible: "Not visible", noRelatedCandidates: "No associated candidates", noCarrierResults: "No carrier results", targetMatching: "Target matching",
    predictedText: "Predicted text", candidateId: "Candidate ID", score: "Score", carrierMissing: "N/A · carrier missing", noMatchResults: "No matching results",
    ocrCandidates: "OCR Candidates", recognizedText: "Recognized text", confidence: "Confidence", candidateStatus: "Status", candidateExtra: "Extra text",
    candidateActive: "Used for matching", candidateFiltered: "Filtered", noOcrCandidates: "No OCR candidates in this frame", frameExtraRatio: "Frame extra-text ratio {ratio} · Extra IDs: {ids}",
    none: "None", carrierDecision: "Carrier decision", targetBatch: "Target matching · Batch {batch}", loadingCall: "Loading call trace...",
    noVlmCalls: "No VLM call traces", fastMatched: "Targets were resolved by fast matching, or no candidates remained for the VLM.", finalSystemPrompt: "Final System Prompt",
    userPayload: "User Payload", finalResponse: "Final Response", allAttempts: "All Attempts ({count})", zoomOut: "Zoom out", zoomIn: "Zoom in",
    resetView: "Reset view", closeViewer: "Close image viewer", imageViewerHint: "Wheel or +/- to zoom · drag to pan · double-click to toggle zoom",
  },
};

const storedLanguage = (() => {
  try { return localStorage.getItem("textvbench-evaluator-language"); } catch (_) { return null; }
})();
let uiLanguage = storedLanguage === "en" || storedLanguage === "zh"
  ? storedLanguage : ((navigator.language || "zh").toLowerCase().startsWith("zh") ? "zh" : "en");
const t = (key, values = {}) => {
  let value = messages[uiLanguage][key] || messages.en[key] || key;
  for (const [name, replacement] of Object.entries(values)) value = value.replaceAll(`{${name}}`, String(replacement));
  return value;
};
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const number = (value, digits = 2) => value == null ? "—" : Number(value).toFixed(digits);
const pct = (value) => value == null ? "—" : `${number(value * 100, 1)}%`;
const json = (value) => JSON.stringify(value ?? {}, null, 2);
const badge = (value, extra = "") => `<span class="badge ${escapeHtml(extra)}">${escapeHtml(value || "—")}</span>`;
const statusLabel = (status) => t({ complete: "statusComplete", error: "statusError" }[status] || status);
const option = (value, label, current) => `<option value="${escapeHtml(value)}" ${value === current ? "selected" : ""}>${escapeHtml(label)}</option>`;

// Relative URLs keep requests inside a VSCode/CloudML forwarding prefix.
const caseUrl = (item) => `api/cases/${encodeURIComponent(item.benchmark)}/${encodeURIComponent(item.case_id)}`;
const assetUrl = (item, folder, filename) => `media/cases/${encodeURIComponent(item.benchmark)}/${encodeURIComponent(item.case_id)}/frames/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}`;

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).error || message; } catch (_) { /* non-JSON error */ }
    throw new Error(message);
  }
  return response.json();
}

function applyStaticLanguage() {
  document.documentElement.lang = uiLanguage === "zh" ? "zh-CN" : "en";
  document.title = t("documentTitle");
  document.querySelector("#product-label").textContent = t("productName");
  document.querySelector("#header-label").textContent = t("headerLabel");
  const switcher = document.querySelector("#language-switch");
  switcher.setAttribute("aria-label", t("languageControl"));
  switcher.querySelectorAll("button").forEach((button) => {
    const active = button.dataset.uiLanguage === uiLanguage;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  updateViewerLabels();
}
function setLanguage(language) {
  if (!messages[language] || language === uiLanguage) return;
  uiLanguage = language;
  try { localStorage.setItem("textvbench-evaluator-language", language); } catch (_) { /* storage unavailable */ }
  applyStaticLanguage();
  route();
}
function showError(error) { app.innerHTML = `<div class="notice error">${escapeHtml(t("readFailed", { error: error.message }))}</div>`; }

function listShell() {
  app.innerHTML = `<div class="page-heading"><div><div class="overline">${t("reviewBundles")}</div><h1>${t("evaluationCases")}</h1></div><button id="refresh" class="command" type="button">${t("refresh")}</button></div>
    <div class="metrics" id="overview-metrics"></div><div class="list-head"><h2>${t("caseList")}</h2><span class="subtle" id="total-label"></span></div>
    <div class="toolbar" role="search"><input id="search" type="search" aria-label="${t("searchCases")}" placeholder="${t("searchPlaceholder")}" value="${escapeHtml(filters.search)}">
      <select id="benchmark" aria-label="${t("benchmark")}">${option("", t("allBenchmarks"), filters.benchmark)}</select><select id="language" aria-label="${t("language")}">${option("", t("allLanguages"), filters.language)}</select>
      <select id="status" aria-label="${t("status")}">${option("", t("allStatuses"), filters.status)}${option("complete", t("statusComplete"), filters.status)}${option("error", t("statusError"), filters.status)}</select>
      <select id="sort" aria-label="${t("sort")}">${option("score_asc", t("scoreAscending"), filters.sort)}${option("score_desc", t("scoreDescending"), filters.sort)}${option("extra_desc", t("extraDescending"), filters.sort)}${option("id", t("sampleIdSort"), filters.sort)}</select></div>
    <div id="table-region" class="table-wrap"><div class="empty">${t("loadingCases")}</div></div>
    <div class="pager"><button id="prev" class="command" type="button">${t("previous")}</button><span id="page-label"></span><button id="next" class="command" type="button">${t("next")}</button></div>`;
  document.querySelector("#refresh").addEventListener("click", () => loadCases(true));
  document.querySelector("#search").addEventListener("input", (event) => { filters.search = event.target.value; filters.page = 1; clearTimeout(searchTimer); searchTimer = setTimeout(loadCases, 250); });
  ["benchmark", "language", "status", "sort"].forEach((field) => document.querySelector(`#${field}`).addEventListener("change", (event) => { filters[field] = event.target.value; filters.page = 1; loadCases(); }));
  document.querySelector("#prev").addEventListener("click", () => { filters.page--; loadCases(); });
  document.querySelector("#next").addEventListener("click", () => { filters.page++; loadCases(); });
  document.querySelector("#table-region").addEventListener("click", (event) => { const row = event.target.closest("tr[data-index]"); if (row) openCase(listing.items[Number(row.dataset.index)]); });
  document.querySelector("#table-region").addEventListener("keydown", (event) => { const row = event.target.closest("tr[data-index]"); if (row && ["Enter", " "].includes(event.key)) { event.preventDefault(); openCase(listing.items[Number(row.dataset.index)]); } });
}

function listParams() {
  const params = new URLSearchParams({ page: filters.page, page_size: 20, sort: filters.sort });
  ["search", "benchmark", "language", "status"].forEach((key) => { if (filters[key]) params.set(key, filters[key]); });
  return params;
}

function renderListing(data) {
  if (!document.querySelector("#table-region")) return;
  for (const [field, values] of [["benchmark", data.facets.benchmarks], ["language", data.facets.languages]]) {
    document.querySelector(`#${field}`).innerHTML = option("", field === "benchmark" ? t("allBenchmarks") : t("allLanguages"), filters[field]) + values.map((value) => option(value, value.toUpperCase(), filters[field])).join("");
  }
  document.querySelector("#overview-metrics").innerHTML = [[t("cases"), data.counts.cases], [t("averageScore"), number(data.counts.average_score, 1)], [t("belowSixty"), data.counts.low_score], [t("evaluationErrors"), data.counts.error]].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
  document.querySelector("#total-label").textContent = t("matchedCount", { count: data.total });
  document.querySelector("#table-region").innerHTML = data.items.length ? `<table class="case-table"><thead><tr><th>${t("sampleId")}</th><th>${t("language")}</th><th>${t("textVScore")}</th><th>${t("spatial")}</th><th>${t("temporal")}</th><th>${t("extraText")}</th><th>${t("sampledFrames")}</th><th>${t("status")}</th></tr></thead><tbody>${data.items.map((item, index) => `<tr tabindex="0" data-index="${index}" aria-label="${escapeHtml(t("viewCase", { id: item.sample_id }))}"><td><div class="case-name">${escapeHtml(item.sample_id)}</div><span class="subtle">${escapeHtml(item.benchmark)} · ${escapeHtml(item.pair_id || "—")}</span></td><td>${escapeHtml((item.language || "—").toUpperCase())}</td><td class="score ${item.metrics.textv_score < 60 ? "low" : ""}">${number(item.metrics.textv_score, 1)}</td><td>${pct(item.metrics.spatial_fidelity)}</td><td>${pct(item.metrics.temporal_fidelity)}</td><td>${pct(item.metrics.extra_text_ratio)}</td><td>${item.frame_count}</td><td>${badge(statusLabel(item.status), item.status)}${item.source_qc && item.source_qc !== "PASS" ? `<br><span class="subtle">${t("sourceQc")}: ${escapeHtml(item.source_qc)}</span>` : ""}</td></tr>`).join("")}</tbody></table>` : `<div class="empty">${t("noCases")}</div>`;
  document.querySelector("#page-label").textContent = t("pageCount", { page: data.page, total: Math.max(1, Math.ceil(data.total / data.page_size)) });
  document.querySelector("#prev").disabled = data.page === 1;
  document.querySelector("#next").disabled = data.page * data.page_size >= data.total;
}

async function loadCases(forceRefresh = false) {
  const request = ++requestNumber;
  const params = listParams();
  const key = params.toString();
  if (forceRefresh) params.set("refresh", "1");
  try {
    const data = await getJson(`api/cases?${params}`);
    if (request !== requestNumber || !document.querySelector("#table-region")) return;
    listing = data;
    listingKey = key;
    renderListing(data);
  } catch (error) { if (request === requestNumber && document.querySelector("#table-region")) document.querySelector("#table-region").innerHTML = `<div class="notice error">${escapeHtml(t("readFailed", { error: error.message }))}</div>`; }
}

function renderList() {
  detail = null;
  ++requestNumber;
  listShell();
  if (listing && listingKey === listParams().toString()) {
    renderListing(listing);
    requestAnimationFrame(() => window.scrollTo(0, listScrollY));
  } else {
    loadCases();
  }
}
function openCase(item) {
  listScrollY = window.scrollY;
  history.replaceState({ ...(history.state || {}), view: "list" }, "", location.href);
  history.pushState({ view: "case", fromList: true }, "", `?case=${encodeURIComponent(`${item.benchmark}/${item.case_id}`)}`);
  renderCase(item.benchmark, item.case_id);
}
function backToList() {
  if (history.state?.fromList) {
    history.back();
  } else {
    history.replaceState({ view: "list" }, "", location.pathname);
    renderList();
  }
}
function metric(label, value) { return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`; }
function targetTable(data) {
  const scored = new Map((data.result.metrics?.targets || []).map((item) => [item.slot_id, item]));
  return `<div class="table-wrap"><table class="table-mini"><thead><tr><th>${t("target")}</th><th>${t("textCarrier")}</th><th>${t("spatial")}</th><th>${t("temporal")}</th></tr></thead><tbody>${(data.manifest.targets || []).map((target) => { const score = scored.get(target.slot_id) || {}; return `<tr><td class="mono">${escapeHtml(target.slot_id)}</td><td><strong>${escapeHtml(target.text)}</strong><br><span class="subtle">${escapeHtml(target.carrier || target.carrier_id)} · ${escapeHtml(target.position)}</span></td><td>${pct(score.spatial)}</td><td>${pct(score.temporal)}</td></tr>`; }).join("") || `<tr><td colspan="4" class="empty-cell">${t("noTargets")}</td></tr>`}</tbody></table></div>`;
}

async function renderCase(benchmark, caseId) {
  const request = ++requestNumber; app.innerHTML = `<div class="empty">${t("loadingEvidence")}</div>`;
  try {
    const data = await getJson(`api/cases/${encodeURIComponent(benchmark)}/${encodeURIComponent(caseId)}`); if (request !== requestNumber) return;
    detail = data; selectedFrame = 0; const s = data.summary;
    app.innerHTML = `<button id="back" class="text-command" type="button">← ${t("backToCases")}</button><div class="page-heading"><div><div class="overline">${escapeHtml(s.benchmark.toUpperCase())} / ${escapeHtml((s.language || "—").toUpperCase())}</div><h1>${escapeHtml(s.sample_id)}</h1><div class="detail-meta"><span>${escapeHtml(s.pair_id || "")}</span><span>${number(s.duration_sec)} s</span><span>${t("framesCount", { count: s.frame_count })}</span><span>${t("sourceQc")}: ${escapeHtml(s.source_qc || "—")}</span></div></div>${badge(statusLabel(s.status), s.status)}</div><div class="metrics">${metric(t("textVScore"), number(s.metrics.textv_score, 1))}${metric(t("spatialFidelity"), pct(s.metrics.spatial_fidelity))}${metric(t("temporalFidelity"), pct(s.metrics.temporal_fidelity))}${metric(t("extraTextRatio"), pct(s.metrics.extra_text_ratio))}</div><div class="detail-grid"><div><section class="section"><div class="section-head"><h2>${t("videoAndFrames")}</h2><span class="subtle">${s.video_available ? t("videoAvailable") : t("videoUnavailable")}</span></div>${s.video_available ? `<video class="video" id="video" controls preload="metadata" src="${caseUrl(s).replace("api/", "media/")}/video"></video>` : ""}<div class="frames" id="frames" aria-label="${t("sampledFrames")}">${data.frames.map((frame, index) => `<button class="frame-button ${index === 0 ? "active" : ""}" type="button" data-index="${index}" aria-label="${escapeHtml(t("viewFrame", { time: number(frame.timestamp, 3) }))}"><img loading="lazy" alt="" src="${assetUrl(s, frame.folder, "source_frame.jpg")}"><span>${number(frame.timestamp, 3)} s · #${frame.frame_index}</span></button>`).join("")}</div><div id="frame-focus"></div></section><section class="section"><h2>${t("rawData")}</h2><div class="toolbar"><select id="raw-choice" aria-label="${t("rawDataType")}">${option("manifest", t("manifest"), "manifest")}${option("input", t("inputRecord"), "manifest")}${option("result", t("fullResult"), "manifest")}${option("config", t("publicConfig"), "manifest")}${option("frame", t("currentFrameResult"), "manifest")}</select></div><pre id="raw-json"></pre></section></div><div><section class="section"><h2>${t("targetsAndScores")}</h2>${targetTable(data)}</section><section class="section"><h2>${t("currentFrameDecision")}</h2><div id="evidence"></div></section><section class="section"><h2>${t("vlmCalls")}</h2><div id="calls"></div></section></div></div>`;
    document.querySelector("#back").addEventListener("click", backToList);
    document.querySelector("#frames").addEventListener("click", (event) => { const button = event.target.closest("button[data-index]"); if (button) renderFrame(Number(button.dataset.index)); });
    document.querySelector("#frame-focus").addEventListener("click", (event) => { const button = event.target.closest("button[data-image-src]"); if (button) openImageViewer(button.dataset.imageSrc, button.dataset.imageTitle); });
    document.querySelector("#raw-choice").addEventListener("change", renderRaw);
    if (data.frames.length) renderFrame(0); else document.querySelector("#frame-focus").innerHTML = `<div class="empty">${t("noFrames")}</div>`;
  } catch (error) { if (request === requestNumber) showError(error); }
}

function renderRaw() {
  if (!detail) return;
  const { manifest, input_record: input, result, config, frames } = detail;
  const fullResult = { ...result, frames: frames.map(({ folder, files, ...frame }) => frame) };
  const sources = { manifest, input, result: fullResult, config, frame: frames[selectedFrame] || {} };
  document.querySelector("#raw-json").textContent = json(sources[document.querySelector("#raw-choice").value]);
}

function renderFrame(index) {
  const data = detail; if (!data || !data.frames[index]) return;
  selectedFrame = index; const frame = data.frames[index]; const s = data.summary; const files = new Set(frame.files);
  document.querySelectorAll(".frame-button").forEach((button, i) => button.classList.toggle("active", i === index));
  const visual = (file, title, purpose) => { const src = assetUrl(s, frame.folder, file); return `<figure class="artifact-figure"><button class="image-open" type="button" data-image-src="${src}" data-image-title="${escapeHtml(title)}" aria-label="${escapeHtml(t("openImage", { title }))}"><img src="${src}" alt="${escapeHtml(title)}, ${number(frame.timestamp, 3)} s"></button><figcaption><strong>${escapeHtml(title)}</strong><span>${escapeHtml(purpose)}</span></figcaption></figure>`; };
  const matching = frame.files.filter((file) => /^matching_marker_batch_\d+\.jpg$/.test(file));
  document.querySelector("#frame-focus").innerHTML = `<div class="section-head"><h3>${t("timeAndFrame", { time: number(frame.timestamp, 3), frame: frame.frame_index })}</h3><span class="subtle">${index + 1} / ${data.frames.length}</span></div><div class="frame-visuals">${files.has("source_frame.jpg") ? visual("source_frame.jpg", t("sourceFrame"), t("sourceFramePurpose")) : ""}${files.has("carrier_marker.jpg") ? visual("carrier_marker.jpg", t("carrierMarker"), t("carrierMarkerPurpose")) : ""}${matching.map((file) => visual(file, t("matchingMarker"), t("matchingMarkerPurpose", { batch: file.match(/\d+/)?.[0] || "0" }))).join("")}</div>`;
  const video = document.querySelector("#video"); if (video && Number.isFinite(frame.timestamp)) try { video.currentTime = frame.timestamp; } catch (_) { /* metadata not ready */ }
  const active = new Set((frame.active_candidates || []).map((candidate) => candidate.id)); const extra = new Set(frame.extra_ids || []);
  document.querySelector("#evidence").innerHTML = `<div class="evidence-heading"><h3>${t("carrier")}</h3>${(frame.carriers || []).map((carrier) => `<span class="pill">${escapeHtml(carrier.carrier_id)}: ${carrier.present ? t("visible") : t("notVisible")} · ${escapeHtml((carrier.candidate_ids || []).join(", ") || t("noRelatedCandidates"))}</span>`).join("") || `<span class="subtle">${t("noCarrierResults")}</span>`}</div><h3>${t("targetMatching")}</h3><div class="table-wrap"><table class="table-mini"><thead><tr><th>${t("slot")}</th><th>${t("predictedText")}</th><th>${t("candidateId")}</th><th>${t("score")}</th></tr></thead><tbody>${(frame.matches || []).map((match) => `<tr><td class="mono">${escapeHtml(match.slot_id)}</td><td>${escapeHtml(match.pred_text || "—")}</td><td class="mono">${escapeHtml((match.selected_ids || []).join(", ") || "—")}</td><td>${match.score == null ? t("carrierMissing") : pct(match.score)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty-cell">${t("noMatchResults")}</td></tr>`}</tbody></table></div><h3 class="evidence-subhead">${t("ocrCandidates")}</h3><div class="table-wrap"><table class="table-mini"><thead><tr><th>ID</th><th>${t("recognizedText")}</th><th>${t("confidence")}</th><th>${t("candidateStatus")}</th></tr></thead><tbody>${(frame.ocr_candidates || []).map((candidate) => `<tr><td class="mono">${escapeHtml(candidate.id)}</td><td>${escapeHtml(candidate.ocr_text)}</td><td>${pct(candidate.confidence)}</td><td class="${extra.has(candidate.id) ? "candidate-extra" : active.has(candidate.id) ? "candidate-active" : ""}">${extra.has(candidate.id) ? t("candidateExtra") : active.has(candidate.id) ? t("candidateActive") : t("candidateFiltered")}</td></tr>`).join("") || `<tr><td colspan="4" class="empty-cell">${t("noOcrCandidates")}</td></tr>`}</tbody></table></div><p class="evidence-note">${t("frameExtraRatio", { ratio: pct(frame.extra_text_ratio), ids: escapeHtml((frame.extra_ids || []).join(", ") || t("none")) })}</p>`;
  const traces = frame.files.filter((file) => /^(?:carrier_vlm_call|matching_vlm_call_batch_\d+)\.json$/.test(file));
  document.querySelector("#calls").innerHTML = traces.length ? traces.map((file, traceIndex) => `<details class="fold" data-trace="${escapeHtml(file)}" data-key="${index}-${traceIndex}"><summary>${file.startsWith("carrier") ? t("carrierDecision") : t("targetBatch", { batch: file.match(/\d+/)[0] })}</summary><div class="fold-body">${t("loadingCall")}</div></details>`).join("") : `<p class="subtle">${t("noVlmCalls")}</p>`;
  if (!traces.some((file) => file.startsWith("matching"))) document.querySelector("#calls").insertAdjacentHTML("beforeend", `<p class="evidence-note">${t("fastMatched")}</p>`);
  document.querySelectorAll(".fold[data-trace]").forEach((fold) => fold.addEventListener("toggle", async () => {
    if (!fold.open || fold.dataset.loaded || fold.dataset.loading) return; fold.dataset.loading = "true"; const body = fold.querySelector(".fold-body");
    try { const trace = await getJson(`${caseUrl(s)}/frames/${encodeURIComponent(frame.folder)}/${encodeURIComponent(fold.dataset.trace)}`); if (detail !== data || selectedFrame !== index) return; body.innerHTML = `${trace.error ? `<p class="trace-error">${escapeHtml(trace.error)}</p>` : ""}<div class="trace-block"><label>${t("finalSystemPrompt")}</label><pre>${escapeHtml(trace.final_prompt || "—")}</pre></div><div class="trace-block"><label>${t("userPayload")}</label><pre>${escapeHtml(json(trace.user_payload))}</pre></div><div class="trace-block"><label>${t("finalResponse")}</label><pre>${escapeHtml(json(trace.final_response))}</pre></div><div class="trace-block"><label>${t("allAttempts", { count: (trace.attempts || []).length })}</label><pre>${escapeHtml(json(trace.attempts))}</pre></div>`; fold.dataset.loaded = "true"; } catch (error) { body.textContent = t("readFailed", { error: error.message }); } finally { delete fold.dataset.loading; }
  }));
  renderRaw();
}

const viewer = document.querySelector("#image-viewer"); const viewerImage = document.querySelector("#viewer-image"); const viewerViewport = document.querySelector("#viewer-viewport");
const MIN_VIEWER_SCALE = 0.1;
const viewerState = { scale: 1, x: 0, y: 0, dragging: false, pointerX: 0, pointerY: 0 };
function updateViewerLabels() {
  if (!viewer) return;
  [["#viewer-zoom-out", "zoomOut"], ["#viewer-zoom-in", "zoomIn"], ["#viewer-reset", "resetView"], ["#viewer-close", "closeViewer"]].forEach(([selector, key]) => { const button = document.querySelector(selector); button.title = t(key); button.setAttribute("aria-label", t(key)); });
  document.querySelector("#viewer-hint").textContent = t("imageViewerHint");
}
function renderViewerTransform() { viewerImage.style.transform = `translate3d(${viewerState.x}px, ${viewerState.y}px, 0) scale(${viewerState.scale})`; document.querySelector("#viewer-scale").textContent = `${Math.round(viewerState.scale * 100)}%`; viewerViewport.classList.toggle("can-pan", viewerState.scale > 1); }
function resetViewer() { Object.assign(viewerState, { scale: 1, x: 0, y: 0, dragging: false }); renderViewerTransform(); }
function setViewerZoom(nextScale, clientX, clientY) {
  const oldScale = viewerState.scale; const scale = Math.min(8, Math.max(MIN_VIEWER_SCALE, nextScale)); if (scale === oldScale) return;
  const rect = viewerViewport.getBoundingClientRect(); const focusX = Number.isFinite(clientX) ? clientX - rect.left - rect.width / 2 : 0; const focusY = Number.isFinite(clientY) ? clientY - rect.top - rect.height / 2 : 0;
  viewerState.x = focusX - ((focusX - viewerState.x) / oldScale) * scale; viewerState.y = focusY - ((focusY - viewerState.y) / oldScale) * scale; viewerState.scale = scale; if (scale <= 1) viewerState.x = viewerState.y = 0; renderViewerTransform();
}
function openImageViewer(src, title) { viewerImage.src = src; viewerImage.alt = title; document.querySelector("#viewer-title").textContent = title; resetViewer(); viewer.showModal(); }

document.querySelector("#language-switch").addEventListener("click", (event) => { const button = event.target.closest("button[data-ui-language]"); if (button) setLanguage(button.dataset.uiLanguage); });
document.querySelector("#viewer-close").addEventListener("click", () => viewer.close());
document.querySelector("#viewer-zoom-in").addEventListener("click", () => setViewerZoom(viewerState.scale * 1.35));
document.querySelector("#viewer-zoom-out").addEventListener("click", () => setViewerZoom(viewerState.scale / 1.35));
document.querySelector("#viewer-reset").addEventListener("click", resetViewer);
viewer.addEventListener("close", () => { viewerImage.removeAttribute("src"); resetViewer(); });
viewer.addEventListener("click", (event) => { if (event.target === viewer) viewer.close(); });
viewerViewport.addEventListener("wheel", (event) => { event.preventDefault(); setViewerZoom(viewerState.scale * (event.deltaY < 0 ? 1.18 : 1 / 1.18), event.clientX, event.clientY); }, { passive: false });
viewerViewport.addEventListener("dblclick", (event) => setViewerZoom(viewerState.scale > 1 ? 1 : 2.5, event.clientX, event.clientY));
viewerViewport.addEventListener("pointerdown", (event) => { if (viewerState.scale <= 1) return; viewerState.dragging = true; viewerState.pointerX = event.clientX; viewerState.pointerY = event.clientY; viewerViewport.setPointerCapture(event.pointerId); viewerViewport.classList.add("dragging"); });
viewerViewport.addEventListener("pointermove", (event) => { if (!viewerState.dragging) return; viewerState.x += event.clientX - viewerState.pointerX; viewerState.y += event.clientY - viewerState.pointerY; viewerState.pointerX = event.clientX; viewerState.pointerY = event.clientY; renderViewerTransform(); });
const endViewerDrag = () => { viewerState.dragging = false; viewerViewport.classList.remove("dragging"); };
viewerViewport.addEventListener("pointerup", endViewerDrag); viewerViewport.addEventListener("pointercancel", endViewerDrag);
viewer.addEventListener("keydown", (event) => { if (["+", "="].includes(event.key)) { event.preventDefault(); setViewerZoom(viewerState.scale * 1.35); } if (event.key === "-") { event.preventDefault(); setViewerZoom(viewerState.scale / 1.35); } if (event.key === "0") { event.preventDefault(); resetViewer(); } });

function route() { const value = new URLSearchParams(location.search).get("case"); const match = value?.match(/^([A-Za-z0-9_][A-Za-z0-9_.-]*)\/([A-Za-z0-9_][A-Za-z0-9_.-]*)$/); if (match) renderCase(match[1], match[2]); else renderList(); }
window.addEventListener("popstate", route); applyStaticLanguage(); route();
