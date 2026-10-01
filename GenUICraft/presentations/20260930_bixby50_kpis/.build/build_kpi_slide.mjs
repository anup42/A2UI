import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const SKILL_DIR = 'C:/Users/anupk/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const RUNTIME_PYTHON = 'C:/Users/anupk/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = 'C:/Users/anupk/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE = process.execPath;
process.env.RUNTIME_PYTHON = RUNTIME_PYTHON;
const workspaceDir = 'C:/Users/anupk/Documents/git/A2UI/GenUICraft/presentations/20260930_bixby50_kpis';
const buildDir = path.join(workspaceDir, '.build');
const outputDir = path.join(workspaceDir, 'output');
const sourcePath = 'C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260929_bixby50_full_gpu_fp32/prefill_2p1s_latency_estimate/summary.json';
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
const family = resolvePresentationFont({ fontFamily: 'Arial', availableFonts: ['Arial'] });
const source = JSON.parse(await fs.readFile(sourcePath, 'utf8'));
const off = source.populations.litert_mtp_off.all_50;
const on = source.populations.litert_mtp_on.all_50;
const fmt = (n) => n.toFixed(2);
const scores = [85.90, 83.26, 80.89];
const throughputRatio = on.decode_speed_supplied_tps / off.decode_speed_supplied_tps;
const scoreDifference = scores[1] - scores[2];
if (fmt(off.estimated_warm_conversion_s) !== '19.84' || fmt(on.estimated_warm_conversion_s) !== '13.19') throw new Error('Latency source mismatch');

const p = Presentation.create({ slideSize: { width: 1600, height: 900 } });
const s = p.slides.add();
s.background.fill = '#FFFFFF';
const C = { ink: '#15243B', muted: '#607084', off: '#235BDC', on: '#007F83', ref: '#8E9BAC', faint: '#EEF8F7' };

function text(name, value, x, y, w, h, size, color = C.ink, bold = false, alignment = 'left') {
  const shape = s.shapes.add({ name, geometry: 'textbox', position: { left: x, top: y, width: w, height: h }, fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: family, fontSize: size, bold, color, alignment, verticalAlignment: 'middle', autoFit: 'none', wrap: 'none', insets: { left: 0, right: 0, top: 0, bottom: 0 } };
  return shape;
}

text('title', 'Gemma 4 E2B on Bixby50', 72, 47, 1456, 70, 56, C.ink, true);
text('subtitle', '50 cases per mode with the trained R64 W4 LiteRT model on GPU FP32', 74, 127, 1450, 40, 26, C.muted);
text('quality-heading', 'UI representation quality', 74, 208, 710, 42, 30, C.ink, true);
text('quality-unit', 'v5.4 score /100, LiteRT results after SDK repair', 74, 253, 710, 34, 22, C.muted);

const chart = s.charts.add('bar', {
  position: { left: 62, top: 300, width: 704, height: 367 },
  categories: ['FP32 checkpoint', 'MTP off', 'MTP on'],
  series: [{
    name: 'v5.4 quality score', values: scores, valuesFormatCode: '0.00', fill: C.off,
    points: [{ idx: 0, fill: C.ref }, { idx: 1, fill: C.off }, { idx: 2, fill: C.on }],
    dataLabelOverrides: scores.map((value, idx) => ({ idx, text: value.toFixed(2), position: 'outEnd', showValue: true, textStyle: { typeface: family, fontSize: 34, bold: true, fill: C.ink } })),
  }],
  barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 95, varyColors: true },
  hasLegend: false,
  xAxis: { visible: true, textStyle: { typeface: family, fontSize: 25, fill: C.ink }, line: { fill: '#DDE3EB', width: 1 }, majorGridlines: null },
  yAxis: { visible: true, min: 0, max: 100, majorUnit: 50, numberFormatCode: '0', textStyle: { typeface: family, fontSize: 19, fill: C.muted }, line: { fill: 'none', width: 0 }, majorGridlines: { fill: '#E8EDF2', width: 1 } },
  dataLabels: { showValue: true, position: 'outEnd', textStyle: { typeface: family, fontSize: 34, bold: true, fill: C.ink } },
  chartFill: '#FFFFFF', chartLine: { fill: 'none', width: 0 }, plotAreaFill: '#FFFFFF', plotAreaLine: { fill: 'none', width: 0 },
});
applyPresentationChartFont(chart, { fontFamily: family });

text('latency-heading', 'Conversion KPIs', 830, 208, 698, 42, 30, C.ink, true);
text('latency-subtitle', 'Estimated latency with the model already loaded', 830, 253, 700, 34, 22, C.muted);
const values = [
  ['Metric', 'MTP off', 'MTP on'],
  ['Decode speed (tokens/s)', '35', '52'],
  ['Average output tokens', fmt(off.output_tokens_mean), fmt(on.output_tokens_mean)],
  ['Decode time (s)', fmt(off.estimated_decode_s), fmt(on.estimated_decode_s)],
  ['Prefill (s)', '2.10', '2.10'],
  ['Other overhead (s)', fmt(off.measured_provider_residual_s + off.measured_converter_and_recording_s), fmt(on.measured_provider_residual_s + on.measured_converter_and_recording_s)],
  ['Total latency (s)', fmt(off.estimated_warm_conversion_s), fmt(on.estimated_warm_conversion_s)],
];
const table = s.tables.add({ rows: 7, columns: 3, left: 830, top: 301, width: 698, height: 366, columnWidths: [346, 176, 176], values });
table.styleOptions = { headerRow: false, totalRow: false, bandedRows: false, firstColumn: false, lastColumn: false };
table.borders.assign({ style: 'solid', fill: '#E3E9F0', width: 1 });
for (let row = 0; row < 7; row++) {
  table.rows[row].height = row === 6 ? 70 : 49.333333;
  for (let col = 0; col < 3; col++) {
    const cell = table.getCell(row, col);
    cell.fill = row === 6 ? C.faint : '#FFFFFF';
    cell.text.style = {
      typeface: family,
      fontSize: row === 6 ? (col === 0 ? 28 : 38) : (row === 0 ? 25 : 24),
      bold: row === 0 || row === 6,
      color: col === 1 ? C.off : (col === 2 ? C.on : C.ink),
      alignment: col === 0 ? 'left' : 'center',
      verticalAlignment: 'middle', autoFit: 'none',
      insets: { left: 10, right: 10, top: 6, bottom: 6 },
    };
  }
}
text('takeaway', `${throughputRatio.toFixed(2)}× decode throughput with MTP, with a ${scoreDifference.toFixed(2)}-point lower quality score`, 74, 707, 1450, 54, 34, C.ink, true);
text('assumptions', 'Supplied: FP32 score 85.90 and decode speeds 35/52 tokens/s. Assumed prefill: 2.10 s.', 74, 791, 1450, 31, 21, C.muted);
text('scope', 'All-50 averages include stopped attempts. Estimates exclude Perplexity generation and UI rendering.', 74, 828, 1450, 31, 21, C.muted);

s.speakerNotes.textFrame.setText([
  'Sources: user-approved KPI values in this conversation, 29–30 September 2026.',
  `Calculation source: ${sourcePath}`,
  'Detailed report: C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260929_bixby50_full_gpu_fp32/prefill_2p1s_latency_estimate/REPORT.md',
  'FP32 checkpoint reference85.90 is user supplied. LiteRT scores83.26 and80.89 are full50 means after published SDK0.5.6 generated-output repair, including unrecoverable attempts as zero. V5.4 measures source-to-UI representation quality rather than source factuality or a visual acceptance percentage.',
  'Decode35/52 tokens per second and prefill2.1seconds are supplied assumptions. Mean native output tokens across50attempts are615.42 without MTP and568.70 with MTP. Native input tokens average3468.62 including the prompt.',
  'Off estimate:615.42/35 +2.1 +0.16089401368 =19.8443225851seconds. On estimate:568.70/52 +2.1 +0.15302515518 =13.1895636167seconds.',
  'Other overhead includes provider dispatch, converter prompt/compile/repair/attribution, and benchmark recording I/O from mixed-device saved runs. These are modelled warm conversion durations, not freshly measured end-to-end screen latency. Excludes initialization, Perplexity generation, rendering and screenshot/test waits.',
  '52/35=1.485714, displayed as1.49x.83.26−80.89=2.37score points. Different output lengths also affect total latency, so do not attribute the entire latency gap solely to speculative decoding.',
  'The50-case averages include6repetition stops off and7on. Excluding those gives estimated warm durations21.58seconds off and14.55seconds on. The complete evaluation reused27verified earlier generations and collected73new outputs on Flip8 and Fold7.',
].join('\n\n'));

await fs.mkdir(outputDir, { recursive: true });
const candidatePath = path.join(buildDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(path.join(buildDir, 'slide-1.layout.json'), await (await s.export({ format: 'layout' })).text());
await fs.writeFile(path.join(buildDir, 'draft.png'), new Uint8Array(await (await p.export({ slide: s, format: 'png', scale: 1 })).arrayBuffer()));
await fs.writeFile(path.join(buildDir, 'authoring.inspect.ndjson'), (await p.inspect({ kind: 'slide,textbox,shape,table,chart,notes', maxChars: 30000 })).ndjson);

const finalPath = path.join(outputDir, process.env.KPI_FINAL_NAME || 'GenUICraft_Bixby50_KPIs.pptx');
const result = await finalizePresentation({
  workspaceDir, candidatePath, finalPath,
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [1],
  requiredNativeChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: true,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '15240000,8572500', '--validate-bullet-geometry', '--validate-heading-fit', '--require-native-table-slide', '1'],
  fontPolicy: { basis: 'design', families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, path.basename(finalPath) + '.validation.json'),
});
await fs.writeFile(path.join(buildDir, 'finalization-result.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ finalPath, preview: path.join(buildDir, 'draft.png'), result }, null, 2));
