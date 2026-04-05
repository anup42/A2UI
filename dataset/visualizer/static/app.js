const state = {
  runId: null,
  tab: "queries",
  offset: 0,
  limit: 50,
  search: "",
  field: "",
  rendererId: "",
  rendererOutputDir: "rendered"
};

const runSelect = document.getElementById("runSelect");
const rendererSelect = document.getElementById("rendererSelect");
const refreshBtn = document.getElementById("refreshBtn");
const summaryEl = document.getElementById("summary");
const metricsWrap = document.getElementById("metricsWrap");
const intentWrap = document.getElementById("intentWrap");
const tabs = document.querySelectorAll(".tab");
const searchInput = document.getElementById("searchInput");
const fieldInput = document.getElementById("fieldInput");
const limitInput = document.getElementById("limitInput");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const tableWrap = document.getElementById("tableWrap");

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res.json();
}

function showMessage(message) {
  summaryEl.innerHTML = `
    <div class="card">
      <div class="label">Status</div>
      <div class="value">${message}</div>
    </div>
  `;
  tableWrap.innerHTML = `<div class="empty">${message}</div>`;
  if (metricsWrap) {
    metricsWrap.innerHTML = "";
  }
  if (intentWrap) {
    intentWrap.innerHTML = "";
  }
}

function setSummary(payload) {
  const agg = payload.aggregates || {};
  const overall = agg.overall_score;
  summaryEl.innerHTML = `
    <div class="card">
      <div class="label">Queries</div>
      <div class="value">${payload.queries ?? 0}</div>
    </div>
    <div class="card">
      <div class="label">Responses</div>
      <div class="value">${payload.responses ?? 0}</div>
    </div>
    <div class="card">
      <div class="label">GenUICraft</div>
      <div class="value">${payload.genui ?? 0}</div>
    </div>
    <div class="card">
      <div class="label">Stage5</div>
      <div class="value">${payload.stage5 ?? 0}</div>
    </div>
    <div class="card">
      <div class="label">Overall Score</div>
      <div class="value">${overall ?? "-"}</div>
    </div>
  `;
}

function setRendererOptions(renderers) {
  if (!rendererSelect) return;
  const variants = Array.isArray(renderers) && renderers.length > 0
    ? renderers
    : [{ id: "json_render", output_dir: "rendered" }];
  rendererSelect.innerHTML = "";
  variants.forEach((renderer) => {
    const opt = document.createElement("option");
    opt.value = renderer.id;
    const outDir = renderer.output_dir || "rendered";
    opt.dataset.outputDir = outDir;
    opt.textContent = `${renderer.id} (${outDir})`;
    rendererSelect.appendChild(opt);
  });

  const existing = variants.find((item) => item.id === state.rendererId);
  const selected = existing || variants[0];
  state.rendererId = selected.id;
  state.rendererOutputDir = selected.output_dir || "rendered";
  rendererSelect.value = selected.id;
}

function formatValue(value, digits = 3) {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "number") {
    const fixed = value.toFixed(digits);
    return fixed.replace(/\.?0+$/, "");
  }
  return String(value);
}


const METRIC_BANDS = {
  overall_score: [
    { label: "Bad", min: null, max: 32.5, range: "< 32.5" },
    { label: "OK", min: 32.5, max: 60.3, range: "32.5 - 60.3" },
    { label: "Good", min: 60.3, max: 85.8, range: "60.3 - 85.8" },
    { label: "Excellent", min: 85.8, max: null, range: "> 85.8" }
  ],
  schema_valid_strict_rate: [
    { label: "Bad", min: null, max: 0.7, range: "< 0.70" },
    { label: "OK", min: 0.7, max: 0.85, range: "0.70 - 0.85" },
    { label: "Good", min: 0.85, max: 0.95, range: "0.85 - 0.95" },
    { label: "Excellent", min: 0.95, max: null, range: ">= 0.95" }
  ],
  content_coverage_avg: [
    { label: "Bad", min: null, max: 0.6, range: "< 0.60" },
    { label: "OK", min: 0.6, max: 0.75, range: "0.60 - 0.75" },
    { label: "Good", min: 0.75, max: 0.9, range: "0.75 - 0.90" },
    { label: "Excellent", min: 0.9, max: null, range: ">= 0.90" }
  ],
  lint_score_avg: [
    { label: "Bad", min: null, max: 0.7, range: "< 0.70" },
    { label: "OK", min: 0.7, max: 0.85, range: "0.70 - 0.85" },
    { label: "Good", min: 0.85, max: 0.95, range: "0.85 - 0.95" },
    { label: "Excellent", min: 0.95, max: null, range: ">= 0.95" }
  ],
  dup_rate_avg: [
    { label: "Bad", min: 0.35, max: null, range: "> 0.35" },
    { label: "OK", min: 0.2, max: 0.35, range: "0.20 - 0.35" },
    { label: "Good", min: 0.1, max: 0.2, range: "0.10 - 0.20" },
    { label: "Excellent", min: null, max: 0.1, range: "<= 0.10" }
  ],
  render_ok_rate: [
    { label: "Bad", min: null, max: 0.5, range: "< 0.50" },
    { label: "OK", min: 0.5, max: 0.75, range: "0.50 - 0.75" },
    { label: "Good", min: 0.75, max: 0.95, range: "0.75 - 0.95" },
    { label: "Excellent", min: 0.95, max: null, range: ">= 0.95" }
  ],
  component_count_avg: [
    { label: "Bad", min: null, max: 8.0, range: "< 8" },
    { label: "OK", min: 8.0, max: 16.0, range: "8 - 15" },
    { label: "Good", min: 16.0, max: 36.0, range: "16 - 35" },
    { label: "Excellent", min: 36.0, max: null, range: "> 35" }
  ],
  component_count_capped_avg: [
    { label: "Bad", min: null, max: 12.0, range: "< 12" },
    { label: "OK", min: 12.0, max: 24.0, range: "12 - 24" },
    { label: "Good", min: 24.0, max: 36.0, range: "24 - 35" },
    { label: "Excellent", min: 36.0, max: null, range: "> 35" }
  ],
  unique_component_types_avg: [
    { label: "Bad", min: null, max: 3.0, range: "< 3" },
    { label: "OK", min: 3.0, max: 5.0, range: "3 - 4" },
    { label: "Good", min: 5.0, max: 8.0, range: "5 - 7" },
    { label: "Excellent", min: 8.0, max: null, range: ">= 8" }
  ],
  max_tree_depth_avg: [
    { label: "Bad", min: null, max: 2.0, range: "< 2" },
    { label: "OK", min: 2.0, max: 4.0, range: "2 - 3" },
    { label: "Excellent", min: 5.0, max: 7.0, range: "5 - 6" },
    { label: "Good", min: 4.0, max: 8.0, range: "4 or 7" },
    { label: "OK", min: 8.0, max: 10.0, range: "8 - 9" },
    { label: "Bad", min: 10.0, max: null, range: ">= 10" }
  ],
  avg_tree_depth_avg: [
    { label: "Bad", min: null, max: 1.0, range: "< 1.0" },
    { label: "OK", min: 1.0, max: 1.5, range: "1.0 - 1.4" },
    { label: "Good", min: 1.5, max: 2.0, range: "1.5 - 1.9" },
    { label: "Excellent", min: 2.0, max: 3.1, range: "2.0 - 3.0" },
    { label: "Good", min: 3.1, max: 3.6, range: "3.1 - 3.5" },
    { label: "OK", min: 3.6, max: 4.6, range: "3.6 - 4.5" },
    { label: "Bad", min: 4.6, max: null, range: "> 4.5" }
  ],
  container_to_text_ratio_avg: [
    { label: "Bad", min: null, max: 0.2, range: "< 0.20" },
    { label: "OK", min: 0.2, max: 0.5, range: "0.20 - 0.50" },
    { label: "Good", min: 0.5, max: 1.2, range: "0.50 - 1.20" },
    { label: "Excellent", min: 1.2, max: 2.01, range: "1.20 - 2.00" },
    { label: "Good", min: 2.01, max: null, range: "> 2.00" }
  ],
  information_chunking_score_avg: [
    { label: "Bad", min: null, max: 0.3, range: "< 0.30" },
    { label: "OK", min: 0.3, max: 0.5, range: "0.30 - 0.50" },
    { label: "Good", min: 0.5, max: 0.7, range: "0.50 - 0.70" },
    { label: "Excellent", min: 0.7, max: null, range: ">= 0.70" }
  ],
  ui_decomposition_score_avg: [
    { label: "Bad", min: null, max: 0.35, range: "< 0.35" },
    { label: "OK", min: 0.35, max: 0.5, range: "0.35 - 0.50" },
    { label: "Good", min: 0.5, max: 0.65, range: "0.50 - 0.65" },
    { label: "Excellent", min: 0.65, max: null, range: ">= 0.65" }
  ],
  ui_modularity_score_avg: [
    { label: "Bad", min: null, max: 0.08, range: "< 0.08" },
    { label: "OK", min: 0.08, max: 0.15, range: "0.08 - 0.15" },
    { label: "Good", min: 0.15, max: 0.28, range: "0.15 - 0.28" },
    { label: "Excellent", min: 0.28, max: null, range: ">= 0.28" }
  ],
  actionable_elements_avg: [
    { label: "Bad", min: null, max: 1.0, range: "0" },
    { label: "OK", min: 1.0, max: 2.0, range: "1" },
    { label: "Good", min: 2.0, max: 4.0, range: "2 - 3" },
    { label: "Excellent", min: 4.0, max: null, range: ">= 4" }
  ],
  action_coverage_avg: [
    { label: "Bad", min: null, max: 0.5, range: "< 0.50" },
    { label: "OK", min: 0.5, max: 0.7, range: "0.50 - 0.70" },
    { label: "Good", min: 0.7, max: 0.9, range: "0.70 - 0.90" },
    { label: "Excellent", min: 0.9, max: null, range: ">= 0.90" }
  ],
  url_as_text_rate_avg: [
    { label: "Bad", min: 0.4, max: null, range: "> 0.40" },
    { label: "OK", min: 0.2, max: 0.4, range: "0.20 - 0.40" },
    { label: "Good", min: 0.05, max: 0.2, range: "0.05 - 0.20" },
    { label: "Excellent", min: null, max: 0.05, range: "<= 0.05" }
  ],
  table_pattern_detected_rate: [
    { label: "Bad", min: null, max: 0.5, range: "< 0.50" },
    { label: "OK", min: 0.5, max: 0.7, range: "0.50 - 0.70" },
    { label: "Good", min: 0.7, max: 0.9, range: "0.70 - 0.90" },
    { label: "Excellent", min: 0.9, max: null, range: ">= 0.90" }
  ],
  table_cell_coverage_avg: [
    { label: "Bad", min: null, max: 0.3, range: "< 0.30" },
    { label: "OK", min: 0.3, max: 0.5, range: "0.30 - 0.50" },
    { label: "Good", min: 0.5, max: 0.75, range: "0.50 - 0.75" },
    { label: "Excellent", min: 0.75, max: null, range: ">= 0.75" }
  ],
  section_heading_coverage_avg: [
    { label: "Bad", min: null, max: 0.3, range: "< 0.30" },
    { label: "OK", min: 0.3, max: 0.6, range: "0.30 - 0.60" },
    { label: "Good", min: 0.6, max: 0.85, range: "0.60 - 0.85" },
    { label: "Excellent", min: 0.85, max: null, range: ">= 0.85" }
  ],
  markdown_leakage_rate_avg: [
    { label: "Bad", min: 0.35, max: null, range: "> 0.35" },
    { label: "OK", min: 0.2, max: 0.35, range: "0.20 - 0.35" },
    { label: "Good", min: 0.08, max: 0.2, range: "0.08 - 0.20" },
    { label: "Excellent", min: null, max: 0.08, range: "<= 0.08" }
  ],
  intent_expectation_pass_rate: [
    { label: "Bad", min: null, max: 0.4, range: "< 0.40" },
    { label: "OK", min: 0.4, max: 0.65, range: "0.40 - 0.65" },
    { label: "Good", min: 0.65, max: 0.85, range: "0.65 - 0.85" },
    { label: "Excellent", min: 0.85, max: null, range: ">= 0.85" }
  ],
  intent_score_avg: [
    { label: "Bad", min: null, max: 0.4, range: "< 0.40" },
    { label: "OK", min: 0.4, max: 0.65, range: "0.40 - 0.65" },
    { label: "Good", min: 0.65, max: 0.85, range: "0.65 - 0.85" },
    { label: "Excellent", min: 0.85, max: null, range: ">= 0.85" }
  ]
};

function inBand(value, band) {
  const minOk = band.min === null || value >= band.min;
  const maxOk = band.max === null || value < band.max;
  return minOk && maxOk;
}

function getMetricBand(metricKey, rawValue) {
  if (rawValue === null || rawValue === undefined || Number.isNaN(Number(rawValue))) {
    return null;
  }
  const rules = METRIC_BANDS[metricKey];
  if (!rules || !Array.isArray(rules) || rules.length === 0) {
    return null;
  }
  const value = Number(rawValue);
  for (const rule of rules) {
    if (inBand(value, rule)) {
      return {
        label: rule.label,
        range: rule.range,
        className: `band-${String(rule.label).toLowerCase()}`
      };
    }
  }
  return null;
}

function createBandLegend() {
  const legend = document.createElement("div");
  legend.className = "band-legend";
  const items = [
    ["Bad", "band-bad"],
    ["OK", "band-ok"],
    ["Good", "band-good"],
    ["Excellent", "band-excellent"]
  ];
  for (const [text, klass] of items) {
    const chip = document.createElement("span");
    chip.className = `band-chip ${klass}`;
    chip.textContent = text;
    legend.appendChild(chip);
  }
  return legend;
}
function renderMetricGrid(title, entries) {
  const section = document.createElement("div");
  const header = document.createElement("div");
  header.className = "section-title";
  header.textContent = title;
  section.appendChild(header);

  const grid = document.createElement("div");
  grid.className = "metric-grid";
  entries.forEach(([label, value]) => {
    const band = getMetricBand(label, value);
    const card = document.createElement("div");
    card.className = "metric-card";
    const valueHtml = `<div class="metric-value">${formatValue(value)}</div>`;
    const bandHtml = band
      ? `
      <div class="metric-band-row">
        <span class="band-chip ${band.className}">${band.label}</span>
        <span class="band-range">${band.range}</span>
      </div>
    `
      : "";
    card.innerHTML = `<div class="metric-label">${label}</div>${valueHtml}${bandHtml}`;
    grid.appendChild(card);
  });
  section.appendChild(grid);
  return section;
}

function setMetrics(payload) {
  if (!metricsWrap) return;
  const agg = payload.aggregates;
  metricsWrap.innerHTML = "";
  intentWrap.innerHTML = "";
  if (!agg) {
    metricsWrap.innerHTML = `<div class="empty">No aggregate metrics available.</div>`;
    return;
  }

  const coreMetrics = [
    ["overall_score", agg.overall_score],
    ["schema_valid_strict_rate", agg.schema_valid_strict_rate],
    ["content_coverage_avg", agg.content_coverage_avg],
    ["lint_score_avg", agg.lint_score_avg],
    ["dup_rate_avg", agg.dup_rate_avg],
    ["render_ok_rate", agg.render_ok_rate]
  ];

  const uiMetrics = [
    ["component_count_avg", agg.component_count_avg],
    ["component_count_capped_avg", agg.component_count_capped_avg],
    ["unique_component_types_avg", agg.unique_component_types_avg],
    ["max_tree_depth_avg", agg.max_tree_depth_avg],
    ["avg_tree_depth_avg", agg.avg_tree_depth_avg],
    ["container_to_text_ratio_avg", agg.container_to_text_ratio_avg],
    ["information_chunking_score_avg", agg.information_chunking_score_avg],
    ["ui_modularity_score_avg", agg.ui_modularity_score_avg],
    ["ui_decomposition_score_avg", agg.ui_decomposition_score_avg]
  ];

  const actionMetrics = [
    ["actionable_elements_avg", agg.actionable_elements_avg],
    ["action_coverage_avg", agg.action_coverage_avg],
    ["url_as_text_rate_avg", agg.url_as_text_rate_avg],
    ["table_pattern_detected_rate", agg.table_pattern_detected_rate],
    ["table_cell_coverage_avg", agg.table_cell_coverage_avg],
    ["section_heading_coverage_avg", agg.section_heading_coverage_avg],
    ["markdown_leakage_rate_avg", agg.markdown_leakage_rate_avg]
  ];

  const intentMetrics = [
    ["intent_expectation_pass_rate", agg.intent_expectation_pass_rate],
    ["intent_score_avg", agg.intent_score_avg]
  ];

  const perfMetrics = [
    ["latency_ms_avg", agg.latency_ms_avg],
    ["latency_ms_p95", agg.latency_ms_p95],
    ["cost_usd_avg", agg.cost_usd_avg]
  ];

  metricsWrap.appendChild(createBandLegend());
  metricsWrap.appendChild(renderMetricGrid("Core Metrics", coreMetrics));
  metricsWrap.appendChild(renderMetricGrid("UI Structure Metrics", uiMetrics));
  metricsWrap.appendChild(renderMetricGrid("Action & Table Metrics", actionMetrics));
  metricsWrap.appendChild(renderMetricGrid("Intent Metrics", intentMetrics));
  metricsWrap.appendChild(renderMetricGrid("Performance Metrics", perfMetrics));

  const intentStats = agg.intent_stats || null;
  if (!intentStats || Object.keys(intentStats).length === 0) {
    intentWrap.innerHTML = `<div class="empty">No intent stats available.</div>`;
    return;
  }
  const intentHeader = document.createElement("div");
  intentHeader.className = "section-title";
  intentHeader.textContent = "Intent Expectations";
  intentWrap.appendChild(intentHeader);

  const headers = [
    "intent",
    "count",
    "expectation_pass_rate",
    "intent_score_avg",
    "table_expected_rate",
    "table_ok_rate",
    "action_expected_rate",
    "action_ok_rate",
    "section_expected_rate",
    "section_ok_rate"
  ];
  const rows = Object.entries(intentStats).map(([intent, stats]) => [
    intent,
    stats.count,
    formatValue(stats.expectation_pass_rate),
    formatValue(stats.intent_score_avg),
    formatValue(stats.table_expected_rate),
    formatValue(stats.table_ok_rate),
    formatValue(stats.action_expected_rate),
    formatValue(stats.action_ok_rate),
    formatValue(stats.section_expected_rate),
    formatValue(stats.section_ok_rate)
  ]);
  intentWrap.appendChild(createTable(headers, rows));
}

function setRuns(runs) {
  runSelect.innerHTML = "";
  runs.forEach((run) => {
    const opt = document.createElement("option");
    opt.value = run.run_id;
    opt.textContent = `${run.run_id} (Q:${run.queries} R:${run.responses} G:${run.genui} S5:${run.stage5 ?? 0})`;
    runSelect.appendChild(opt);
  });
  if (!state.runId && runs.length > 0) {
    state.runId = runs[0].run_id;
    runSelect.value = state.runId;
  }
}

function createTable(headers, rows) {
  const table = document.createElement("table");
  table.className = "data-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  rows.forEach((cells) => {
    const tr = document.createElement("tr");
    cells.forEach((cell) => {
      const td = document.createElement("td");
      if (cell instanceof Node) {
        td.appendChild(cell);
      } else {
        td.textContent = cell ?? "";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}

function createDetails(label, content) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = label;
  details.appendChild(summary);
  const pre = document.createElement("pre");
  pre.textContent = content;
  details.appendChild(pre);
  return details;
}

function renderQueries(items) {
  const headers = ["query_id", "intent", "difficulty", "tags", "query_text"];
  const rows = items.map((item) => [
    item.query_id,
    item.intent,
    item.difficulty,
    Array.isArray(item.tags) ? item.tags.join(", ") : "",
    item.query_text
  ]);
  tableWrap.innerHTML = "";
  tableWrap.appendChild(createTable(headers, rows));
}

function renderResponses(items) {
  const headers = ["response_id", "query_id", "n_idx", "response_text"];
  const rows = items.map((item) => [
    item.response_id,
    item.query_id,
    item.n_idx,
    item.response_text
  ]);
  tableWrap.innerHTML = "";
  tableWrap.appendChild(createTable(headers, rows));
}

function renderA2ui(items) {
  const headers = ["ui_id", "response_id", "intent", "validation", "errors", "IR", "render"];
  const rows = items.map((item) => {
    const validation = item.validation || {};
    const valText = `parse:${validation.json_parse_ok} strict:${validation.schema_valid_strict}`;
    const errors = Array.isArray(validation.errors) && validation.errors.length
      ? validation.errors[0]
      : "";
    const intent = item.intent || item.intent_bucket || "";
    const irPayload = item.genui_json || item.a2ui_json || item.toon || "";
    const irText = typeof irPayload === "string"
      ? irPayload
      : JSON.stringify(irPayload, null, 2);
    const renderLink = document.createElement("a");
    renderLink.href = `/runs/${state.runId}/${state.rendererOutputDir}/${item.ui_id}.html`;
    renderLink.textContent = "html";
    renderLink.target = "_blank";
    const pngLink = document.createElement("a");
    pngLink.href = `/runs/${state.runId}/${state.rendererOutputDir}/${item.ui_id}.png`;
    pngLink.textContent = "png";
    pngLink.target = "_blank";
    const container = document.createElement("span");
    container.appendChild(renderLink);
    container.appendChild(document.createTextNode(" | "));
    container.appendChild(pngLink);
    const errorCell = errors ? createDetails("view", errors) : "";
    const irCell = irText ? createDetails("view", irText) : "";
    return [item.ui_id, item.response_id, intent, valText, errorCell, irCell, container];
  });
  tableWrap.innerHTML = "";
  tableWrap.appendChild(createTable(headers, rows));
}

function renderStage5(items) {
  const headers = ["response_id", "query_id", "created_at", "render_error", "output"];
  const rows = items.map((item) => {
    const renderInfo = item.render || {};
    const renderError = renderInfo.error || "";
    const createdAt = item.created_at || "";

    const htmlLink = document.createElement("a");
    htmlLink.href = `/runs/${state.runId}/${item.html_path}`;
    htmlLink.textContent = "html";
    htmlLink.target = "_blank";

    const container = document.createElement("span");
    container.appendChild(htmlLink);

    if (item.image_path) {
      const pngLink = document.createElement("a");
      pngLink.href = `/runs/${state.runId}/${item.image_path}`;
      pngLink.textContent = "png";
      pngLink.target = "_blank";
      container.appendChild(document.createTextNode(" | "));
      container.appendChild(pngLink);
    }

    return [item.response_id, item.query_id, createdAt, renderError, container];
  });
  tableWrap.innerHTML = "";
  tableWrap.appendChild(createTable(headers, rows));
}

async function loadRuns() {
  const payload = await fetchJson("/api/runs");
  setRuns(payload.runs || []);
  if (!payload.runs || payload.runs.length === 0) {
    showMessage("No runs found. Check runs directory or server path.");
  }
}

async function loadSummary() {
  if (!state.runId) return;
  const payload = await fetchJson(`/api/run/${state.runId}/summary`);
  setSummary(payload);
  setRendererOptions(payload.renderers || []);
  setMetrics(payload);
}

async function loadTab() {
  if (!state.runId) return;
  const params = new URLSearchParams({
    offset: String(state.offset),
    limit: String(state.limit),
    search: state.search,
    field: state.field
  });
  const payload = await fetchJson(`/api/run/${state.runId}/${state.tab}?${params}`);
  const items = payload.items || [];
  if (state.tab === "queries") {
    renderQueries(items);
  } else if (state.tab === "responses") {
    renderResponses(items);
  } else if (state.tab === "stage5") {
    renderStage5(items);
  } else {
    renderA2ui(items);
  }
}

function resetPaging() {
  state.offset = 0;
}

runSelect.addEventListener("change", async () => {
  state.runId = runSelect.value;
  resetPaging();
  await loadSummary();
  await loadTab();
});

if (rendererSelect) {
  rendererSelect.addEventListener("change", async () => {
    const selected = rendererSelect.options[rendererSelect.selectedIndex];
    state.rendererId = rendererSelect.value;
    state.rendererOutputDir = (selected && selected.dataset.outputDir) || "rendered";
    if (state.tab === "genui") {
      await loadTab();
    }
  });
}

refreshBtn.addEventListener("click", async () => {
  await loadRuns();
  await loadSummary();
  await loadTab();
});

tabs.forEach((tab) => {
  tab.addEventListener("click", async () => {
    tabs.forEach((btn) => btn.classList.remove("active"));
    tab.classList.add("active");
    state.tab = tab.dataset.tab;
    resetPaging();
    await loadTab();
  });
});

searchInput.addEventListener("change", async () => {
  state.search = searchInput.value.trim();
  resetPaging();
  await loadTab();
});

fieldInput.addEventListener("change", async () => {
  state.field = fieldInput.value.trim();
  resetPaging();
  await loadTab();
});

limitInput.addEventListener("change", async () => {
  state.limit = Number(limitInput.value) || 50;
  resetPaging();
  await loadTab();
});

prevBtn.addEventListener("click", async () => {
  state.offset = Math.max(0, state.offset - state.limit);
  await loadTab();
});

nextBtn.addEventListener("click", async () => {
  state.offset += state.limit;
  await loadTab();
});

(async function init() {
  try {
    await loadRuns();
    await loadSummary();
    await loadTab();
  } catch (err) {
    showMessage(`Failed to load data: ${err}`);
  }
})();






