const state = {
  runId: null,
  tab: "queries",
  offset: 0,
  limit: 50,
  search: "",
  field: ""
};

const runSelect = document.getElementById("runSelect");
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
      <div class="label">Overall Score</div>
      <div class="value">${overall ?? "-"}</div>
    </div>
  `;
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

function renderMetricGrid(title, entries) {
  const section = document.createElement("div");
  const header = document.createElement("div");
  header.className = "section-title";
  header.textContent = title;
  section.appendChild(header);

  const grid = document.createElement("div");
  grid.className = "metric-grid";
  entries.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="metric-label">${label}</div>
      <div class="metric-value">${formatValue(value)}</div>
    `;
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
    ["unique_component_types_avg", agg.unique_component_types_avg],
    ["max_tree_depth_avg", agg.max_tree_depth_avg],
    ["avg_tree_depth_avg", agg.avg_tree_depth_avg],
    ["container_to_text_ratio_avg", agg.container_to_text_ratio_avg],
    ["information_chunking_score_avg", agg.information_chunking_score_avg],
    ["ui_modularity_score_avg", agg.ui_modularity_score_avg]
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
    opt.textContent = `${run.run_id} (Q:${run.queries} R:${run.responses} G:${run.genui})`;
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
  const headers = ["ui_id", "response_id", "validation", "errors", "render"];
  const rows = items.map((item) => {
    const validation = item.validation || {};
    const valText = `parse:${validation.json_parse_ok} strict:${validation.schema_valid_strict}`;
    const errors = Array.isArray(validation.errors) && validation.errors.length
      ? validation.errors[0]
      : "";
    const renderLink = document.createElement("a");
    renderLink.href = `/runs/${state.runId}/rendered/${item.ui_id}.html`;
    renderLink.textContent = "html";
    renderLink.target = "_blank";
    const pngLink = document.createElement("a");
    pngLink.href = `/runs/${state.runId}/rendered/${item.ui_id}.png`;
    pngLink.textContent = "png";
    pngLink.target = "_blank";
    const container = document.createElement("span");
    container.appendChild(renderLink);
    container.appendChild(document.createTextNode(" | "));
    container.appendChild(pngLink);
    const errorCell = errors ? createDetails("view", errors) : "";
    return [item.ui_id, item.response_id, valText, errorCell, container];
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

