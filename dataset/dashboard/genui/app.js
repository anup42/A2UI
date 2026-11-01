"use strict";
const $ = id => document.getElementById(id);
let state = null, detail = null, tab = "source", pending = false, listSignature = "", categorySignature = "";
let catalog = null, catalogLoading = false, catalogAttempted = false, setupRun = "";
const number = value => value == null ? "—" : Number(value).toLocaleString();
const label = value => String(value || "").replaceAll("_", " ");
const text = (id, value) => { $(id).textContent = value; };
function node(tag, value, className) { const el = document.createElement(tag); el.textContent = value; if (className) el.className = className; return el; }
function reasonLabel(reason) {
  const labels={content_unit_fidelity:"Source content missing",exact_numbers_dates_units_fbeta:"Exact values differ",unsupported_external_action:"Unbound action",action_and_source_link_fidelity:"Action/reference mismatch","semantic_role_difference:formula":"Formula differs"};
  return [...new Set(reason.split(";").filter(v=>!["training_not_eligible","canonical_training_ineligible"].includes(v)).map(v=>{v=v.replace(/^review:|^fidelity:/,"");return labels[v]||label(v);} ))].join(" · ") || "Training eligibility failed";
}
async function api(path, body) {
  const response = await fetch("/api/" + path, body === undefined ? {cache: "no-store"} : {method: "POST", headers: {"Content-Type": "application/json", "X-GenUI-Token": state.csrf}, body: JSON.stringify(body)});
  const result = await response.json(); if (!response.ok) throw new Error(result.error || "Request failed"); return result;
}
function error(message) { $("error").hidden = !message; text("error", message || ""); }
async function action(name) {
  if (pending || !state) return;
  pending = true;
  try {
    const result = await api(name, {});
    if (result.authUrl || result.verificationUrl) {
      const url = new URL(result.authUrl || result.verificationUrl);
      if (url.protocol !== "https:" || !["auth.openai.com", "chatgpt.com"].includes(url.hostname)) throw new Error("Unexpected sign-in address");
      $("login-link").href = url.href; text("device-code", result.userCode ? "Code: " + result.userCode : ""); $("login-flow").hidden = false;
    }
    error(null); await refresh();
  } catch (exc) { error(exc.message); } finally { pending = false; }
}
function render(s) {
  renderSetup(s);
  text("connection", "● Local evidence"); $("connection").classList.add("good");
  text("run-name", s.name); text("run-meta", [s.config.model, label(s.config.selection_profile), s.config.pilot_profile, s.transport === "chatgpt" ? "ChatGPT / Codex" : "OpenAI Batch"].filter(Boolean).join("  ·  "));
  text("run-status", label(s.status)); $("run-status").classList.toggle("good", s.status === "validated");
  const finished = s.completed + s.provider_failed + s.skipped;
  $("progress-bar").style.width = (s.requested ? Math.min(100, finished / s.requested * 100) : 0) + "%";
  text("progress-text", `${number(s.completed)} of ${number(s.requested)} generated` + (s.active_custom_id ? " · generating next source" : "") + (s.uncertain ? ` · ${s.uncertain} uncertain` : ""));
  text("elapsed", s.started_at ? `Elapsed ${Math.floor(((s.updated_at || Date.now()/1000)-s.started_at)/60)} min` : "");
  text("prepared", number(s.requested)); text("generated", number(s.completed)); text("accepted", number(s.accepted)); text("held", number(s.held));
  text("generated-sub", `${s.provider_failed} API failures · ${s.uncertain} uncertain · ${s.skipped} skipped`);
  text("accepted-sub", s.accepted == null ? "Awaiting strict validation" : `${(100*s.accepted/Math.max(1,s.requested)).toFixed(1)}% acceptance · pilot only`);
  const chatgpt = s.transport === "chatgpt";
  $("start").disabled = !chatgpt || !s.auth.signed_in || s.busy || s.sealed || s.uncertain > 0;
  $("collect").disabled = !chatgpt || s.busy || s.sealed || s.completed+s.skipped !== s.requested;
  text("start", s.completed ? "Resume remaining sources" : "Start generation");
  text("transport-note", chatgpt ? "Uses your Codex allowance. Each source runs once; validation follows." : "API Batch archive · read-only local status. Prepare a separate GenUI run for ChatGPT sign-in.");
  text("billing-mode", chatgpt ? "Subscription" : "API Batch");
  text("auth-state", s.auth.signed_in ? `Connected to GenUI · ChatGPT ${s.auth.plan || "account"}` : s.auth.checked ? (s.auth.error || "No ChatGPT account connected to GenUI") : "Check your GenUI account to connect.");
  for (const id of ["signin", "device"]) $(id).disabled = s.busy;
  if (s.auth.signed_in) $("login-flow").hidden = true;
  const u = s.usage || {}, known = Object.hasOwn(u, "input_tokens") && u.records_without_usage < Math.max(1, s.completed+s.skipped+s.uncertain);
  for (const [id, key] of [["input-tokens","input_tokens"],["output-tokens","output_tokens"],["cached-tokens","cached_input_tokens"],["reasoning-tokens","reasoning_tokens"]]) text(id, number(known ? u[key] : null));
  text("cost-label", chatgpt ? "Billing" : "Cost from reported usage");
  text("cost", chatgpt ? "Account allowance" : u.batch_cost_usd_from_reported_tokens_estimated == null ? "Not available" : "$"+u.batch_cost_usd_from_reported_tokens_estimated.toFixed(4));
  text("usage-note", chatgpt ? "No API dollar charge is inferred. Reasoning is included in output. Token counts are only those reported by Codex; no guaranteed cache hit or API-style output cap." : "Estimated from reported Batch usage, including rejected output and cache writes. Reasoning is included in output; taxes and regional premiums excluded.");
  if (u.records_without_usage) text("usage-note", $("usage-note").textContent + ` Usage unavailable for ${u.records_without_usage} attempted records.`);
  if (u.records_without_complete_cache_usage) text("usage-note", $("usage-note").textContent + " Cache usage is incomplete; cost cannot be calculated reliably.");
  const limits = s.auth.limits || {};
  text("rate-limits", [limits.primary, limits.secondary].filter(Boolean).map(l => `${l.windowDurationMins || "?"}-minute window: ${l.usedPercent}% used`).join(" · "));
  const categoryValue = $("category").value, newCategories = JSON.stringify(Object.keys(s.categories));
  const rebuildCategories = newCategories !== categorySignature;
  if(rebuildCategories) { $("category").replaceChildren(node("option","All categories")); $("category").firstChild.value=""; categorySignature=newCategories; }
  $("categories").replaceChildren();
  for (const [key,c] of Object.entries(s.categories)) {
    const row = node("div","","category-row"), track = node("div","","bar-track"), fill = node("div","","bar-fill");
    fill.style.width = `${c.total ? 100*c.accepted/c.total : 0}%`; track.append(fill);
    row.append(node("span",label(key)),track,node("b",`${s.accepted == null ? "—" : c.accepted}/${c.total}`)); $("categories").append(row);
    if(rebuildCategories) { const opt = node("option",label(key)); opt.value = key; $("category").append(opt); }
  }
  $("category").value = categoryValue;
  const gaps = s.selection.numbered_citations;
  text("citation-gaps", gaps ? `Citation coverage: ${gaps.selected_heuristic_candidates ?? "—"} of ${gaps.requested ?? "—"} requested numbered-citation sources found. ${gaps.reference_slots_replaced_by_url_sources ?? 0} slots use URL-bearing sources. Citation candidates are heuristic, not independently verified.` : "Golden/Bixby and validation sources are excluded at preparation. References and source fidelity remain mandatory.");
  $("reasons").replaceChildren();
  const reasons = Object.entries(s.rejection_reasons).sort((a,b)=>b[1]-a[1]);
  if (!reasons.length) $("reasons").append(node("span",s.accepted == null ? "Quality decisions appear after validation." : "No rejection reasons recorded.","small"));
  for (const [reason,count] of reasons.slice(0,5)) { const el=node("span",`${count} · ${reasonLabel(reason)}`,"reason"); el.title=reason; $("reasons").append(el); }
  $("rescore-panel").hidden = !s.rescore;
  if (s.rescore) text("rescore-panel", `Separate offline reassessment: ${s.rescore.v2_accepted}/${s.requested} accepted with fidelity_v2, versus ${s.rescore.official_accepted}/${s.requested} in the original report. Existing labels: ${s.rescore.baseline_v2_accepted}/${s.requested}. The original decisions below are unchanged.`);
  text("refresh-time", "Local files checked " + new Date(s.local_snapshot_at*1000).toLocaleTimeString() + (chatgpt ? "" : " · no remote API polling"));
  if (s.error) error(s.error);
  renderSamples();
}
function fillEfforts(preferred) {
  const model = catalog?.models.find(m => m.model === $("generation-model").value);
  $("generation-effort").replaceChildren();
  for (const effort of model?.reasoning_efforts || []) {
    const option = node("option", label(effort)); option.value = effort; $("generation-effort").append(option);
  }
  const available = model?.reasoning_efforts || [];
  $("generation-effort").value = available.includes(preferred) ? preferred : available.includes(model?.default_reasoning_effort) ? model.default_reasoning_effort : available[0] || "";
}
async function loadModels(force = false) {
  if (catalogLoading) return;
  catalogLoading = true; catalogAttempted = true;
  text("model-status", "Loading models from your signed-in account…");
  try {
    const selected = $("generation-model").value || state?.config.model, effort = $("generation-effort").value || state?.config.reasoning_effort;
    catalog = await api(force ? "models/refresh" : "models", force ? {} : undefined);
    $("generation-model").replaceChildren();
    for (const model of catalog.models) { const opt = node("option", model.display_name || model.model); opt.value = model.model; $("generation-model").append(opt); }
    $("generation-model").value = catalog.models.some(m => m.model === selected) ? selected : (catalog.models.find(m => m.is_default) || catalog.models[0]).model;
    fillEfforts(effort);
    text("model-status", `${catalog.models.length} account models · ${catalog.codex_version} · checked ${new Date(catalog.refreshed_at*1000).toLocaleTimeString()}`);
  } catch (exc) { catalog = null; text("model-status", "Model list unavailable: " + exc.message); }
  finally { catalogLoading = false; if (state) renderSetup(state); }
}
function renderSetup(s) {
  text("header-auth", s.auth.signed_in ? `ChatGPT ${s.auth.plan || "account"} connected ↓` : "Sign in with ChatGPT ↓");
  if (setupRun !== s.path) {
    setupRun = s.path; detail = null; listSignature = "";
    const d = s.setup_defaults || {};
    for (const [id,key] of [["samples","samples"],["seed","seed"],["profile","selection_profile"],["summary","reasoning_summary"],["timeout","timeout"],["input","input_dir"],["output","output_dir"]]) {
      if (d[key] != null) $("generation-"+id).value = d[key];
    }
    $("sample-content").textContent = "Choose a source to inspect its original text and generated output.";
    text("sample-title", "Select a sample"); text("sample-badge", ""); text("sample-reasons", "");
  }
  if (!s.auth.signed_in) { catalog = null; catalogAttempted = false; text("model-status", "Connect your ChatGPT account to load available models."); }
  $("refresh-models").disabled = !s.auth.signed_in || s.busy || catalogLoading;
  $("setup-fields").disabled = !s.auth.signed_in || s.busy || !catalog || catalogLoading;
  $("prepare").disabled = $("setup-fields").disabled;
  text("prepare-status", s.job === "prepare" ? "Preparing sources and checking fidelity locally. This may take several minutes…" : s.transport === "chatgpt" && s.status === "prepared" ? "Run prepared. Review its model and settings below, then click Start generation when ready." : "");
  if (s.auth.signed_in && !catalogAttempted) loadModels();
}
async function prepareRun(event) {
  event.preventDefault();
  if (pending || !catalog || !state || state.busy) return;
  const options = {model:$("generation-model").value, reasoning_effort:$("generation-effort").value,
    reasoning_summary:$("generation-summary").value, samples:Number($("generation-samples").value),
    seed:Number($("generation-seed").value), timeout:Number($("generation-timeout").value),
    selection_profile:$("generation-profile").value, input_dir:$("generation-input").value, output_dir:$("generation-output").value};
  if (options.selection_profile === "failure_mix_v1" && ![100,1000].includes(options.samples)) { error("Failure mix v1 requires 100 or 1,000 samples. Choose Random for another size."); return; }
  pending = true; $("prepare").disabled = true;
  try { await api("prepare", options); error(null); await refresh(); }
  catch (exc) { error(exc.message); }
  finally { pending = false; if (state) renderSetup(state); }
}
function renderSamples() {
  if (!state) return;
  const search = $("search").value.toLowerCase(), category=$("category").value, status=$("status").value;
  const rows=state.samples.filter(s=>(!category||s.category===category)&&(!status||s.status===status)&&(!search||(s.source_id+" "+s.reasons.join(" ")).toLowerCase().includes(search)));
  text("sample-count", `${rows.length} matching sources`);
  const signature=JSON.stringify(rows)+":"+detail?.index;
  if(signature===listSignature) return;
  listSignature=signature;
  const scrollTop=$("sample-list").scrollTop;
  $("sample-list").replaceChildren();
  for (const s of rows) {
    const button=node("button","","sample-row"+(detail?.index===s.index?" selected":""));
    button.append(node("span",s.source_id),node("small",`${label(s.category)} · ${s.status}`));
    button.onclick=async()=>{try {detail=await api("sample?index="+s.index); showDetail(); renderSamples();}catch(exc){error(exc.message);}};
    $("sample-list").append(button);
  }
  $("sample-list").scrollTop=scrollTop;
}
function showDetail(){if(!detail)return;text("sample-title",detail.source_id);text("sample-badge",detail.status);text("sample-reasons",detail.reasons.join(" · "));text("sample-content",detail[tab] || (tab==="output"?"No final output available for this source.":"No source text available."));$("source-tab").setAttribute("aria-pressed",String(tab==="source"));$("output-tab").setAttribute("aria-pressed",String(tab==="output"));}
async function refresh(){try{state=await api("state");render(state);}catch(exc){text("connection","Disconnected");$("connection").classList.remove("good");error(exc.message);}}
$("signin").onclick=()=>action("auth/login"); $("device").onclick=()=>action("auth/device"); $("check-auth").onclick=()=>action("auth/check"); $("start").onclick=()=>action("run"); $("collect").onclick=()=>action("collect");
$("refresh-models").onclick=()=>loadModels(true);
$("generation-model").onchange=()=>fillEfforts();
$("prepare-form").onsubmit=prepareRun;
for(const id of ["search","category","status"]) $(id).addEventListener("input",renderSamples);
$("source-tab").onclick=()=>{tab="source";showDetail();};$("output-tab").onclick=()=>{tab="output";showDetail();};
async function loop(){await refresh();setTimeout(loop,3000);}loop();
