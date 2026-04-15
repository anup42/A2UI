import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { stitch } from "@google/stitch-sdk";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..", "..");

function utcStamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    d.getUTCFullYear() +
    pad(d.getUTCMonth() + 1) +
    pad(d.getUTCDate()) +
    "_" +
    pad(d.getUTCHours()) +
    pad(d.getUTCMinutes()) +
    pad(d.getUTCSeconds())
  );
}

function parseArgs(argv) {
  const out = {
    runId: "subset10_g3pro_iconcatalog_20260216_020032",
    limit: -1,
    modelId: "GEMINI_3_PRO",
    responseId: "",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if ((a === "--run-id" || a === "--run_id") && argv[i + 1]) {
      out.runId = argv[i + 1];
      i += 1;
    } else if (a === "--limit" && argv[i + 1]) {
      out.limit = Number(argv[i + 1]);
      i += 1;
    } else if ((a === "--model-id" || a === "--model_id") && argv[i + 1]) {
      out.modelId = argv[i + 1];
      i += 1;
    } else if ((a === "--response-id" || a === "--response_id") && argv[i + 1]) {
      out.responseId = argv[i + 1];
      i += 1;
    }
  }
  return out;
}

function safeBaseName(value) {
  return String(value || "item")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .slice(0, 120);
}

function buildPrompt(record) {
  const content = record.response_text || "";
  return [
    "Create a polished mobile UI screen from the content below.",
    "Use clear visual hierarchy, cards/sections, and concise spacing.",
    "Preserve factual content and links from the source text.",
    "",
    "CONTENT START",
    content,
    "CONTENT END",
  ].join("\n");
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

async function readJsonl(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line, idx) => {
      try {
        return JSON.parse(line);
      } catch (err) {
        throw new Error(`Failed to parse JSONL line ${idx + 1}: ${err.message}`);
      }
    });
}

async function tryDownload(url, destPath) {
  if (!url) return { ok: false, error: "empty_url" };
  try {
    const res = await fetch(url);
    if (!res.ok) return { ok: false, error: `http_${res.status}` };
    const ctype = (res.headers.get("content-type") || "").toLowerCase();
    if (ctype.includes("text") || ctype.includes("json") || ctype.includes("html")) {
      const text = await res.text();
      await fs.writeFile(destPath, text, "utf-8");
      return { ok: true };
    }
    const arr = await res.arrayBuffer();
    await fs.writeFile(destPath, Buffer.from(arr));
    return { ok: true };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

async function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractScreenDataFromRaw(raw) {
  const candidates = [];
  if (raw && typeof raw === "object") {
    if (raw.design?.screens) candidates.push(raw.design.screens);
    if (Array.isArray(raw.outputComponents)) {
      for (const comp of raw.outputComponents) {
        if (comp?.design?.screens) candidates.push(comp.design.screens);
        if (comp?.screens) candidates.push(comp.screens);
      }
    }
    if (raw.screens) candidates.push(raw.screens);
  }
  for (const maybeScreens of candidates) {
    if (Array.isArray(maybeScreens) && maybeScreens.length > 0) {
      const first = maybeScreens[0];
      if (first && typeof first === "object") return first;
    }
  }
  return null;
}

function toScreenId(screenData) {
  if (!screenData || typeof screenData !== "object") return "";
  if (screenData.id) return String(screenData.id);
  if (screenData.screenId) return String(screenData.screenId);
  if (screenData.name && String(screenData.name).includes("/screens/")) {
    return String(screenData.name).split("/screens/").pop() || "";
  }
  return "";
}

async function generateScreenPayload(project, prompt, modelId, maxAttempts = 3) {
  let lastError = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const screen = await project.generate(prompt, "MOBILE", modelId);
      const htmlUrl = await screen.getHtml();
      const imageUrl = await screen.getImage();
      return {
        screenId: screen.screenId || screen.id || "",
        htmlUrl: htmlUrl || "",
        imageUrl: imageUrl || "",
        via: "project.generate",
      };
    } catch (err) {
      lastError = err;
      if (attempt < maxAttempts) {
        await delay(1200 * attempt);
      }
    }
  }
  try {
    const raw = await stitch.callTool("generate_screen_from_text", {
      projectId: project.projectId,
      prompt,
      deviceType: "MOBILE",
      modelId,
    });
    const screenData = extractScreenDataFromRaw(raw);
    const screenId = toScreenId(screenData);
    let htmlUrl = screenData?.htmlCode?.downloadUrl || "";
    let imageUrl = screenData?.screenshot?.downloadUrl || "";
    if ((!htmlUrl || !imageUrl) && screenId) {
      const fetched = await project.getScreen(screenId);
      if (!htmlUrl) htmlUrl = (await fetched.getHtml()) || "";
      if (!imageUrl) imageUrl = (await fetched.getImage()) || "";
    }
    if (!screenId && !htmlUrl && !imageUrl) {
      const fallbackErr = new Error(
        "Unable to parse screen output from generate_screen_from_text fallback path."
      );
      fallbackErr.raw = raw;
      throw fallbackErr;
    }
    return { screenId, htmlUrl, imageUrl, via: "callTool_fallback" };
  } catch (fallbackErr) {
    if (lastError) {
      fallbackErr.previous_error = lastError.message || String(lastError);
    }
    throw fallbackErr;
  }
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (!process.env.STITCH_API_KEY && !process.env.STITCH_ACCESS_TOKEN) {
    throw new Error("Missing Stitch auth. Set STITCH_API_KEY or STITCH_ACCESS_TOKEN.");
  }

  const runDir = path.join(REPO_ROOT, "dataset", "data", "runs", opts.runId);
  const responsesPath = path.join(runDir, "responses.jsonl");
  const responses = await readJsonl(responsesPath);
  let selected = opts.limit > 0 ? responses.slice(0, opts.limit) : responses;
  if (opts.responseId) {
    selected = selected.filter((r) => String(r.response_id || "") === opts.responseId);
  }
  if (!selected.length) throw new Error("No responses found to process.");

  const stamp = utcStamp();
  const outDir = path.join(REPO_ROOT, "stitch", "outputs", `${opts.runId}_stitch_${stamp}`);
  const htmlDir = path.join(outDir, "html");
  const imageDir = path.join(outDir, "images");
  await ensureDir(outDir);
  await ensureDir(htmlDir);
  await ensureDir(imageDir);

  const projectTitle = `A2UI ${opts.runId} ${stamp}`;
  console.log(`[info] Creating Stitch project: ${projectTitle}`);
  const project = await stitch.createProject(projectTitle);
  console.log(`[info] projectId=${project.projectId}`);

  const results = [];
  for (let i = 0; i < selected.length; i += 1) {
    const record = selected[i];
    const responseId = record.response_id || `row_${i + 1}`;
    const queryId = record.query_id || "";
    const prompt = buildPrompt(record);
    const itemPrefix = `${String(i + 1).padStart(2, "0")}_${safeBaseName(responseId)}`;
    const startedAt = Date.now();

    console.log(`[info] [${i + 1}/${selected.length}] generating ${responseId}`);
    try {
      const generated = await generateScreenPayload(project, prompt, opts.modelId, 3);
      const htmlUrl = generated.htmlUrl;
      const imageUrl = generated.imageUrl;
      const htmlPath = path.join(htmlDir, `${itemPrefix}.html`);
      const imagePath = path.join(imageDir, `${itemPrefix}.png`);
      const htmlDownload = await tryDownload(htmlUrl, htmlPath);
      const imageDownload = await tryDownload(imageUrl, imagePath);

      results.push({
        idx: i + 1,
        query_id: queryId,
        response_id: responseId,
        project_id: project.projectId,
        screen_id: generated.screenId || null,
        html_url: htmlUrl,
        image_url: imageUrl,
        generated_via: generated.via,
        html_saved: htmlDownload.ok,
        image_saved: imageDownload.ok,
        html_error: htmlDownload.ok ? null : htmlDownload.error,
        image_error: imageDownload.ok ? null : imageDownload.error,
        latency_ms: Date.now() - startedAt,
        status: "ok",
      });
    } catch (err) {
      results.push({
        idx: i + 1,
        query_id: queryId,
        response_id: responseId,
        project_id: project.projectId,
        screen_id: null,
        html_url: null,
        image_url: null,
        html_saved: false,
        image_saved: false,
        html_error: null,
        image_error: null,
        latency_ms: Date.now() - startedAt,
        status: "error",
        error: err.message,
      });
      console.error(`[error] ${responseId}: ${err.message}`);
    }
  }

  const manifest = {
    created_at: new Date().toISOString(),
    run_id: opts.runId,
    model_id: opts.modelId,
    project_id: project.projectId,
    project_title: projectTitle,
    total: selected.length,
    success: results.filter((r) => r.status === "ok").length,
    failed: results.filter((r) => r.status !== "ok").length,
    output_dir: path.relative(REPO_ROOT, outDir).replace(/\\/g, "/"),
    records: results,
  };
  const manifestPath = path.join(outDir, "stitch_manifest.json");
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf-8");

  const jsonlPath = path.join(outDir, "stitch_manifest.jsonl");
  await fs.writeFile(
    jsonlPath,
    results.map((r) => JSON.stringify(r)).join("\n") + "\n",
    "utf-8"
  );

  console.log(`[done] Manifest: ${manifestPath}`);
  console.log(`[done] Generated ${manifest.success}/${manifest.total} screens`);
}

main().catch((err) => {
  console.error(`[fatal] ${err.message}`);
  process.exit(1);
});
