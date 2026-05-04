const {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel,
  text,
  image,
  shape,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} = await import("@oai/artifact-tool");
const fs = await import("node:fs/promises");
const kyotoBytes = await fs.readFile("scratch/assets/kyoto_render_crop.png");
const kyotoRenderDataUrl = `data:image/png;base64,${kyotoBytes.toString("base64")}`;

const presentation = Presentation.create({
  slideSize: { width: 1920, height: 1080 },
});

const slide = presentation.slides.add();

const colors = {
  bg: "#08111F",
  ink: "#F8FAFC",
  muted: "#AAB7CA",
  line: "#2A3A52",
  blue: "#38BDF8",
  green: "#34D399",
  amber: "#FBBF24",
  violet: "#A78BFA",
  panel: "#101B2D",
  panel2: "#0D1727",
  code: "#07101C",
};

const labelStyle = {
  fontSize: 22,
  color: colors.muted,
  bold: true,
  fontFace: "Aptos",
};

const bodyStyle = {
  fontSize: 26,
  color: colors.ink,
  fontFace: "Aptos",
};

const monoStyle = {
  fontSize: 19,
  color: "#DDE7F6",
  fontFace: "Consolas",
};

function pill(label, value, color) {
  return panel(
    {
      name: `pill-${label}`,
      fill: colors.panel2,
      line: { style: "solid", width: 1, fill: colors.line },
      borderRadius: "rounded-full",
      padding: { x: 18, y: 10 },
      width: fill,
      height: hug,
    },
    row(
      { width: fill, height: hug, gap: 12, align: "center" },
      [
        shape({
          name: `dot-${label}`,
          width: fixed(12),
          height: fixed(12),
          fill: color,
          line: { style: "solid", width: 0, fill: color },
          borderRadius: "rounded-full",
        }),
        text(label, {
          name: `label-${label}`,
          width: fixed(120),
          height: hug,
          style: labelStyle,
        }),
        text(value, {
          name: `value-${label}`,
          width: fill,
          height: hug,
          style: { ...bodyStyle, fontSize: 24 },
        }),
      ],
    ),
  );
}

function flowStep(title, body, color) {
  return panel(
    {
      name: `flow-${title}`,
      fill: colors.panel,
      line: { style: "solid", width: 1.5, fill: color },
      borderRadius: "rounded-xl",
      padding: { x: 22, y: 18 },
      width: fill,
      height: fixed(132),
    },
    column(
      { width: fill, height: fill, gap: 8, justify: "center" },
      [
        text(title, {
          name: `flow-title-${title}`,
          width: fill,
          height: hug,
          style: { fontSize: 28, bold: true, color, fontFace: "Aptos" },
        }),
        text(body, {
          name: `flow-body-${title}`,
          width: fill,
          height: hug,
          style: { fontSize: 20, color: colors.muted, fontFace: "Aptos" },
        }),
      ],
    ),
  );
}

slide.compose(
  layers(
    { name: "background", width: fill, height: fill },
    [
      shape({
        name: "bg-field",
        width: fill,
        height: fill,
        fill: colors.bg,
        line: { style: "solid", width: 0, fill: colors.bg },
      }),
      shape({
        name: "left-accent",
        width: fixed(28),
        height: fill,
        fill: colors.blue,
        line: { style: "solid", width: 0, fill: colors.blue },
      }),
    ],
  ),
  { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 },
);

slide.compose(
  grid(
    {
      name: "slide-root",
      width: fill,
      height: fill,
      padding: { x: 78, y: 60 },
      rows: [auto, fr(1), auto],
      columns: [fr(1.05), fr(0.92)],
      columnGap: 48,
      rowGap: 34,
    },
    [
      column(
        {
          name: "title-stack",
          columnSpan: 2,
          width: fill,
          height: hug,
          gap: 14,
        },
        [
          text("GenUICraft FlatSpec IR: data model + UI graph", {
            name: "slide-title",
            width: fill,
            height: hug,
            style: {
              fontSize: 56,
              bold: true,
              color: colors.ink,
              fontFace: "Aptos Display",
            },
          }),
          text("Kyoto response -> FlatSpec JSON -> OneUI-native Android render.", {
            name: "slide-subtitle",
            width: wrap(1500),
            height: hug,
            style: {
              fontSize: 25,
              color: colors.muted,
              fontFace: "Aptos",
            },
          }),
        ],
      ),

      column(
        { name: "left-column", width: fill, height: fill, gap: 18 },
        [
          row(
            { name: "pipeline-row", width: fill, height: hug, gap: 16, align: "center" },
            [
              flowStep("Response", "Natural language itinerary, media, actions", colors.amber),
              text("->", {
                name: "arrow-1",
                width: fixed(36),
                height: hug,
                style: { fontSize: 36, color: colors.muted, bold: true },
              }),
              flowStep("FlatSpec", "root + state + elements", colors.blue),
              text("->", {
                name: "arrow-2",
                width: fixed(36),
                height: hug,
                style: { fontSize: 36, color: colors.muted, bold: true },
              }),
              flowStep("OneUI", "Renderer owns layout, style, actions", colors.green),
            ],
          ),

          column(
            {
              name: "ir-table",
              width: fill,
              height: hug,
              gap: 10,
            },
            [
              pill("root", "Entry point: kyoto_itinerary", colors.blue),
              pill("state", "Reusable data: budgetRows, dayCards", colors.green),
              pill("elements", "Flat component graph: Text, Table, Card, Image, Button", colors.violet),
              pill("bindings", "$item, repeat, statePath, openUrl", colors.amber),
            ],
          ),

          panel(
            {
              name: "code-panel",
              fill: colors.code,
              line: { style: "solid", width: 1.2, fill: colors.line },
              borderRadius: "rounded-xl",
              padding: { x: 24, y: 22 },
              width: fill,
              height: fill,
            },
            column(
              { name: "code-stack", width: fill, height: fill, gap: 12 },
              [
                text("Representative IR pattern", {
                  name: "code-title",
                  width: fill,
                  height: hug,
                  style: { fontSize: 26, bold: true, color: colors.ink, fontFace: "Aptos" },
                }),
                rule({ name: "code-rule", width: fill, stroke: colors.line, weight: 1 }),
                text(`{
  "root": "kyoto_itinerary",
  "state": {
    "budgetRows": [...],
    "dayCards": [{ "title": "...", "image": "...", "url": "..." }]
  },
  "elements": {
    "budgetTable": { "type": "Table",
      "props": { "statePath": "/budgetRows" }},
    "dayCardTemplate": { "type": "Card",
      "repeat": { "statePath": "/dayCards" }}
  }
}`, {
                  name: "code-snippet",
                  width: fill,
                  height: hug,
                  style: monoStyle,
                }),
              ],
            ),
          ),
        ],
      ),

      panel(
        {
          name: "render-panel",
          fill: "#111A2A",
          line: { style: "solid", width: 1.2, fill: colors.line },
          borderRadius: "rounded-2xl",
          padding: { x: 22, y: 22 },
          width: fill,
          height: fill,
        },
        column(
          { name: "render-stack", width: fill, height: fill, gap: 16 },
          [
            row(
              { name: "render-heading-row", width: fill, height: hug, align: "center", justify: "between" },
              [
                text("Rendered Android surface", {
                  name: "render-heading",
                  width: fill,
                  height: hug,
                  style: { fontSize: 28, bold: true, color: colors.ink, fontFace: "Aptos" },
                }),
                text("08_u_000008_01", {
                  name: "sample-id",
                  width: fixed(180),
                  height: hug,
                  style: { fontSize: 16, color: colors.blue, fontFace: "Consolas" },
                }),
              ],
            ),
            image({
              name: "kyoto-render",
              dataUrl: kyotoRenderDataUrl,
              width: fill,
              height: grow(1),
              fit: "contain",
              borderRadius: "rounded-xl",
              alt: "Android screenshot of rendered Kyoto GenUICraft itinerary",
            }),
          ],
        ),
      ),

      row(
        {
          name: "bottom-strip",
          columnSpan: 2,
          width: fill,
          height: hug,
          gap: 22,
          align: "center",
        },
        [
          text("Why this matters", {
            name: "bottom-label",
            width: fixed(210),
            height: hug,
            style: { fontSize: 22, bold: true, color: colors.blue, fontFace: "Aptos" },
          }),
          text("No executable UI code. Compact state-driven tables/cards. Local validation and repair before render. Compose owns OneUI typography, spacing, cards, dark mode, and actions.", {
            name: "bottom-proof",
            width: fill,
            height: hug,
            style: { fontSize: 23, color: colors.ink, fontFace: "Aptos" },
          }),
        ],
      ),
    ],
  ),
  { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 },
);

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save("output/genuicraft_ir_explainer.pptx");

const pngBlob = await presentation.export({ slide, format: "png" });
await fs.writeFile(
  "scratch/genuicraft_ir_explainer_slide1.png",
  Buffer.from(await pngBlob.arrayBuffer()),
);

const layoutBlob = await presentation.export({ slide, format: "layout" });
await fs.writeFile(
  "scratch/genuicraft_ir_explainer_slide1.layout.json",
  Buffer.from(await layoutBlob.arrayBuffer()),
);

console.log(JSON.stringify({
  pptx: "output/genuicraft_ir_explainer.pptx",
  png: "scratch/genuicraft_ir_explainer_slide1.png",
  layout: "scratch/genuicraft_ir_explainer_slide1.layout.json",
}, null, 2));

process.exit(0);
