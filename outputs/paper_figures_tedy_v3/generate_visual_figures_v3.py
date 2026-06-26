import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as pe


OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


COLORS = {
    "ink": "#0f172a",
    "muted": "#526173",
    "line": "#c9d6e5",
    "bg": "#fbfdff",
    "panel": "#ffffff",
    "blue": "#2563eb",
    "blue_soft": "#e8f0ff",
    "teal": "#0891b2",
    "teal_soft": "#e6fbff",
    "purple": "#7c3aed",
    "purple_soft": "#f1eaff",
    "green": "#16a34a",
    "green_soft": "#e9fbea",
    "amber": "#f59e0b",
    "amber_soft": "#fff5d9",
    "rose": "#e11d48",
    "rose_soft": "#ffe8ef",
    "gray_soft": "#f5f7fb",
}


def new_fig(width=16, height=9):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def text(ax, x, y, s, size=12, weight="normal", color=None, ha="left", va="center", **kw):
    family = kw.pop("family", "DejaVu Sans")
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        family=family,
        **kw,
    )


def box(ax, x, y, w, h, label=None, sub=None, color="blue", radius=0.035, lw=1.5, fill=None):
    fc = fill or COLORS[f"{color}_soft"]
    ec = COLORS[color]
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius * min(w, h)}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        path_effects=[pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.09), pe.Normal()],
    )
    ax.add_patch(patch)
    if label:
        text(ax, x + 3, y + h - 5, label, size=13, weight="bold", color=ec)
    if sub:
        text(ax, x + 3, y + h - 11, sub, size=9.3, color=COLORS["muted"], va="top")
    return patch


def chip(ax, x, y, s, color="blue", w=None):
    if w is None:
        w = max(8, 0.48 * len(s) + 4)
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            4.2,
            boxstyle="round,pad=0.006,rounding_size=2",
            facecolor=COLORS[f"{color}_soft"],
            edgecolor=COLORS[color],
            linewidth=1,
        )
    )
    text(ax, x + w / 2, y + 2.1, s, size=7.5, weight="bold", color=COLORS[color], ha="center")
    return w


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.8, rad=0.0):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color or COLORS["muted"],
            lw=lw,
            shrinkA=3,
            shrinkB=3,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def stage_card(ax, x, y, w, h, title, subtitle, color, icon_fn=None):
    box(ax, x, y, w, h, color=color, fill=COLORS[f"{color}_soft"])
    if icon_fn:
        icon_fn(ax, x + 4.0, y + h / 2, color)
        tx = x + 8.2
    else:
        tx = x + 3
    text(ax, tx, y + h - 5.0, title, size=10.2, weight="bold", color=COLORS[color])
    text(ax, tx, y + h - 10.0, subtitle, size=7.9, color=COLORS["muted"], va="top")


def icon_circle(ax, cx, cy, r, color, fill=None):
    ax.add_patch(patches.Circle((cx, cy), r, facecolor=fill or COLORS[f"{color}_soft"], edgecolor=COLORS[color], linewidth=1.4))


def icon_user(ax, cx, cy, color="blue"):
    icon_circle(ax, cx, cy, 3.4, color)
    ax.add_patch(patches.Circle((cx, cy + 1.25), 0.9, facecolor=COLORS[color], edgecolor="none"))
    ax.add_patch(patches.Arc((cx, cy - 1.4), 4, 3.2, theta1=20, theta2=160, color=COLORS[color], lw=2))


def icon_doc(ax, cx, cy, color="teal"):
    icon_circle(ax, cx, cy, 3.4, color)
    ax.add_patch(patches.FancyBboxPatch((cx - 1.25, cy - 1.8), 2.5, 3.8, boxstyle="round,pad=0.02,rounding_size=.25", facecolor="white", edgecolor=COLORS[color], lw=1.2))
    for i in range(3):
        ax.plot([cx - 0.8, cx + 0.8], [cy + 0.8 - i * 0.9, cy + 0.8 - i * 0.9], color=COLORS[color], lw=1)


def icon_ir(ax, cx, cy, color="purple"):
    icon_circle(ax, cx, cy, 3.4, color)
    text(ax, cx, cy + 0.1, "{ }", size=13, weight="bold", color=COLORS[color], ha="center")


def icon_shield(ax, cx, cy, color="green"):
    icon_circle(ax, cx, cy, 3.4, color)
    verts = [(cx, cy + 2), (cx + 1.7, cy + 1.1), (cx + 1.25, cy - 1.5), (cx, cy - 2.25), (cx - 1.25, cy - 1.5), (cx - 1.7, cy + 1.1), (cx, cy + 2)]
    ax.add_patch(patches.Polygon(verts, facecolor="white", edgecolor=COLORS[color], lw=1.3))
    ax.plot([cx - 0.75, cx - 0.18, cx + 0.95], [cy - 0.15, cy - 0.75, cy + 0.65], color=COLORS[color], lw=1.8, solid_capstyle="round")


def icon_phone(ax, cx, cy, color="green", scale=1.0):
    w, h = 5.2 * scale, 9.0 * scale
    ax.add_patch(patches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=1.2", facecolor="white", edgecolor=COLORS[color], lw=1.6))
    ax.add_patch(patches.FancyBboxPatch((cx - w / 2 + 0.55 * scale, cy - h / 2 + 0.8 * scale), w - 1.1 * scale, h - 1.6 * scale, boxstyle="round,pad=0.02,rounding_size=.8", facecolor="#f8fafc", edgecolor=COLORS["line"], lw=0.8))
    for i, c in enumerate(["blue", "green", "purple"]):
        ax.add_patch(patches.FancyBboxPatch((cx - 1.45 * scale, cy + 2.2 * scale - i * 2.2 * scale), 2.9 * scale, 1.2 * scale, boxstyle="round,pad=0.02,rounding_size=.5", facecolor=COLORS[f"{c}_soft"], edgecolor="none"))


def icon_table(ax, x, y, w, h, color="teal"):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=.8", facecolor="white", edgecolor=COLORS[color], lw=1.2))
    ax.add_patch(patches.Rectangle((x, y + h - 3.0), w, 3.0, facecolor=COLORS[f"{color}_soft"], edgecolor="none"))
    for i in range(1, 4):
        ax.plot([x, x + w], [y + i * h / 4, y + i * h / 4], color=COLORS["line"], lw=0.8)
    for i in range(1, 3):
        ax.plot([x + i * w / 3, x + i * w / 3], [y, y + h], color=COLORS["line"], lw=0.8)


def draw_mobile_screen(ax, x, y, w, h, mode="weather"):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=2.8", facecolor="#0f172a", edgecolor="#223047", lw=1.4))
    ax.add_patch(patches.FancyBboxPatch((x + 1.1, y + 1.6), w - 2.2, h - 3.2, boxstyle="round,pad=0.012,rounding_size=2", facecolor="#f8fafc", edgecolor="none"))
    if mode == "weather":
        ax.add_patch(patches.FancyBboxPatch((x + 2.1, y + h - 11), w - 4.2, 7.2, boxstyle="round,pad=0.012,rounding_size=1.3", facecolor="#dff6ff", edgecolor="none"))
        ax.add_patch(patches.Circle((x + 5.2, y + h - 7.3), 1.25, facecolor="#fbbf24", edgecolor="none"))
        text(ax, x + 8, y + h - 6.6, "27°C", size=8, weight="bold")
        for i in range(3):
            ax.add_patch(patches.FancyBboxPatch((x + 2.1, y + h - 18 - i * 6), w - 4.2, 4.5, boxstyle="round,pad=0.01,rounding_size=.9", facecolor="#ffffff", edgecolor=COLORS["line"], lw=0.6))
            ax.add_patch(patches.Circle((x + 4, y + h - 15.7 - i * 6), 0.8, facecolor="#fde68a", edgecolor="none"))
    elif mode == "cards":
        for i, c in enumerate(["blue", "green", "amber"]):
            ax.add_patch(patches.FancyBboxPatch((x + 2, y + h - 10 - i * 7), w - 4, 5.6, boxstyle="round,pad=0.01,rounding_size=1", facecolor=COLORS[f"{c}_soft"], edgecolor=COLORS["line"], lw=0.6))
            ax.add_patch(patches.Circle((x + 4.2, y + h - 7.2 - i * 7), 1.1, facecolor=COLORS[c], edgecolor="none", alpha=.75))
            ax.plot([x + 6.2, x + w - 3], [y + h - 6.3 - i * 7, y + h - 6.3 - i * 7], color=COLORS["muted"], lw=1.2)
            ax.plot([x + 6.2, x + w - 7], [y + h - 8.1 - i * 7, y + h - 8.1 - i * 7], color=COLORS["line"], lw=1.2)
    else:
        icon_table(ax, x + 2, y + h - 18, w - 4, 13, color="purple")


def figure_architecture():
    fig, ax = new_fig()
    text(ax, 4, 94, "TEDY: Agent Response to Native Mobile UI", size=24, weight="bold")
    text(ax, 4, 89.5, "A compact semantic IR lets the renderer own platform fidelity, adaptive layouts, and accessibility.", size=12.5, color=COLORS["muted"])

    stages = [
        ("User query", "natural task", "blue", icon_user),
        ("Response", "structured content", "teal", icon_doc),
        ("Flat-spec IR", "root/state/elements", "purple", icon_ir),
        ("Contract", "validate + repair", "green", icon_shield),
    ]
    x = 5
    for i, (title, subtitle, color, icon) in enumerate(stages):
        stage_card(ax, x, 67, 18, 15, title, subtitle, color, icon)
        if i < len(stages) - 1:
            arrow(ax, x + 18.5, 74.2, x + 23, 74.2)
        x += 23

    box(ax, 6, 35, 38, 23, color="purple", fill="#fcfbff")
    text(ax, 9, 53.8, "Semantic JSON payload", size=12.5, weight="bold", color=COLORS["purple"])
    text(ax, 9, 49.7, "Stable training/evaluation target", size=8.6, color=COLORS["muted"])
    text(ax, 9, 45.0, '{ "root": "main",', size=9.7, color=COLORS["ink"], family="DejaVu Sans Mono")
    text(ax, 9, 40.9, '  "state": { "rows": [...] },', size=9.7, color=COLORS["ink"], family="DejaVu Sans Mono")
    text(ax, 9, 36.8, '  "elements": { "main": {...}, "table": {...} } }', size=9.2, color=COLORS["ink"], family="DejaVu Sans Mono")

    box(ax, 52, 33.5, 30, 24.5, color="green", fill="#fbfffb")
    text(ax, 55, 53.8, "Renderer-owned realization", size=12.5, weight="bold", color=COLORS["green"])
    text(ax, 55, 49.7, "Domain templates + device adaptation", size=8.6, color=COLORS["muted"])
    for j, item in enumerate(["Weather / flight / booking cards", "Portrait vs landscape strategy", "Tables, code, formulas, email", "Sources and row-level actions"]):
        ax.add_patch(patches.Circle((55, 45.5 - j * 3.45), 0.62, facecolor=COLORS["green"], edgecolor="none"))
        text(ax, 57, 45.5 - j * 3.45, item, size=8.1)
    draw_mobile_screen(ax, 86, 34, 9, 26, mode="cards")
    text(ax, 90.5, 31.5, "Native UI", size=10.5, weight="bold", color=COLORS["green"], ha="center")
    arrow(ax, 44, 46.5, 52, 46.5, color=COLORS["green"], lw=2.2)
    arrow(ax, 82, 46.5, 86, 46.5, color=COLORS["green"], lw=2.2)

    chip(ax, 8, 20, "compact", "purple")
    chip(ax, 19, 20, "validatable", "green")
    chip(ax, 33, 20, "mobile-first", "blue")
    chip(ax, 48, 20, "renderer adaptive", "teal")
    chip(ax, 66, 20, "training-ready", "amber")
    save(fig, "fig1_tedy_architecture_v3")


def figure_spec_anatomy():
    fig, ax = new_fig()
    text(ax, 4, 94, "Flat-Spec IR Contract", size=24, weight="bold")
    text(ax, 4, 89.5, "The model emits semantic structure once; compact tables and state keep token cost low while preserving values.", size=12.5, color=COLORS["muted"])

    box(ax, 5, 61, 26, 22, color="blue")
    text(ax, 8, 78, "root", size=12.5, weight="bold", color=COLORS["blue"])
    icon_ir(ax, 10.5, 70.5, "blue")
    text(ax, 16, 71, '"main"', size=16, weight="bold", color=COLORS["blue"], family="DejaVu Sans Mono")
    text(ax, 8, 64.8, "Single entry id.\nRenderer starts from one node.", size=8.2, color=COLORS["muted"], va="top")

    box(ax, 37, 61, 26, 22, color="teal")
    text(ax, 40, 78, "state", size=12.5, weight="bold", color=COLORS["teal"])
    icon_table(ax, 40.5, 68.2, 9.5, 8.2, color="teal")
    text(ax, 52, 73.4, "rows[]", size=12.5, weight="bold", color=COLORS["teal"])
    text(ax, 52, 69.2, "actions{}", size=9.5, color=COLORS["muted"])
    text(ax, 40, 64.8, "Tables stay data-first.\nNo expanded cell trees.", size=8.2, color=COLORS["muted"], va="top")

    box(ax, 69, 61, 26, 22, color="purple")
    text(ax, 72, 78, "elements", size=12.5, weight="bold", color=COLORS["purple"])
    for k, c in enumerate(["Stack", "Card", "Table", "Image"]):
        chip(ax, 72 + (k % 2) * 10, 72.5 - (k // 2) * 5.2, c, "purple", w=8.2)
    text(ax, 72, 64.8, "Flat id → component map.\nDangling refs are caught.", size=8.2, color=COLORS["muted"], va="top")

    arrow(ax, 31.5, 72, 36.5, 72)
    arrow(ax, 63.5, 72, 68.5, 72)

    box(ax, 5, 27, 43, 24, color="teal", fill="#fbffff")
    text(ax, 8, 46.5, "Compact table example", size=12.5, weight="bold", color=COLORS["teal"])
    text(ax, 8, 42.9, "All source cells preserved; renderer chooses presentation.", size=8.4, color=COLORS["muted"])
    icon_table(ax, 9, 31.8, 34, 9.8, color="teal")
    text(ax, 10.4, 40.4, "Day", size=7.0, weight="bold", color=COLORS["teal"])
    text(ax, 21.5, 40.4, "Condition", size=7.0, weight="bold", color=COLORS["teal"])
    text(ax, 33.5, 40.4, "High", size=7.0, weight="bold", color=COLORS["teal"])
    for i, row in enumerate([("Mon", "Cloudy", "28°"), ("Tue", "Rain", "27°"), ("Wed", "Clear", "30°")]):
        yy = 38.8 - i * 2.45
        text(ax, 10.4, yy, row[0], size=7.3)
        text(ax, 21.5, yy, row[1], size=7.3)
        text(ax, 34, yy, row[2], size=7.3)
    chip(ax, 10, 29.5, "Table.props.columns", "teal", w=17)
    chip(ax, 28, 29.5, "statePath=/forecast", "green", w=16)

    box(ax, 54, 27, 41, 24, color="purple", fill="#fcfbff")
    text(ax, 57, 46.5, "Supported semantic components", size=12.5, weight="bold", color=COLORS["purple"])
    text(ax, 57, 42.9, "Small catalog with renderer-owned templates.", size=8.4, color=COLORS["muted"])
    icons = [
        ("Text", "blue"), ("Card", "green"), ("Table", "teal"), ("Image", "amber"),
        ("Button", "purple"), ("Tabs", "rose"), ("Code", "blue"), ("Email", "green"),
    ]
    for idx, (name, color) in enumerate(icons):
        xx = 58.5 + (idx % 4) * 8.6
        yy = 37.8 - (idx // 4) * 5.9
        icon_circle(ax, xx, yy, 2.0, color)
        text(ax, xx, yy - 3.2, name, size=7.2, ha="center", color=COLORS["muted"])
    save(fig, "fig2_tedy_flat_spec_contract_v3")


def figure_adaptive_rendering():
    fig, ax = new_fig()
    text(ax, 4, 94, "Adaptive Rendering From the Same Table IR", size=24, weight="bold")
    text(ax, 4, 89.5, "Tables remain compact in IR; Android selects cards, timelines, or true tables from domain and shape.", size=12.5, color=COLORS["muted"])

    box(ax, 4, 59, 23, 24, color="teal")
    text(ax, 7, 78, "Input IR", size=12.5, weight="bold", color=COLORS["teal"])
    text(ax, 7, 73.6, "single Table component", size=8.8, color=COLORS["muted"])
    icon_table(ax, 8, 64.6, 15, 7.2, color="teal")
    chip(ax, 8, 62.5, "domain", "teal", w=9)
    chip(ax, 18, 62.5, "columns", "green", w=9)

    box(ax, 35, 59, 26, 24, color="purple")
    text(ax, 38, 78, "Shape detector", size=12.5, weight="bold", color=COLORS["purple"])
    text(ax, 38, 73.6, "entity-row • matrix • schedule", size=8.4, color=COLORS["muted"])
    for i, label in enumerate(["weather", "flight", "comparison", "key-value"]):
        chip(ax, 39.5 + (i % 2) * 13.0, 67.5 - (i // 2) * 5.5, label, "purple", w=11.5)
    arrow(ax, 27.5, 71, 34.5, 71)

    box(ax, 69, 59, 26, 24, color="green")
    text(ax, 72, 78, "Presentation policy", size=12.5, weight="bold", color=COLORS["green"])
    text(ax, 72, 73.6, "orientation + width aware", size=8.5, color=COLORS["muted"])
    text(ax, 72, 69.5, "portrait <600dp → cards", size=8.8, weight="bold", color=COLORS["green"])
    text(ax, 72, 65.7, "landscape/tablet → table", size=8.8, weight="bold", color=COLORS["green"])
    text(ax, 72, 61.9, "wide matrix → sticky table", size=8.8, weight="bold", color=COLORS["green"])
    arrow(ax, 61.5, 71, 68.5, 71)

    box(ax, 5, 19, 27, 31, color="blue", fill="#f9fbff")
    text(ax, 8, 45, "Portrait phone", size=12.5, weight="bold", color=COLORS["blue"])
    text(ax, 8, 40.5, "Entity rows become\nreadable cards.", size=8.4, color=COLORS["muted"], va="top")
    text(ax, 8, 27.5, "No clipped columns.\nActions stay row-attached.", size=8.1, color=COLORS["muted"], va="top")
    draw_mobile_screen(ax, 20, 21.5, 8.6, 24.5, mode="cards")

    box(ax, 37, 19, 27, 31, color="teal", fill="#fbffff")
    text(ax, 40, 45, "Landscape / tablet", size=12.5, weight="bold", color=COLORS["teal"])
    text(ax, 40, 40.5, "True table when space is available.", size=8.4, color=COLORS["muted"])
    ax.add_patch(patches.FancyBboxPatch((41, 28.8), 19, 10.2, boxstyle="round,pad=0.012,rounding_size=1.2", facecolor="white", edgecolor=COLORS["teal"], lw=1.5))
    icon_table(ax, 42.5, 30, 16, 8.0, color="teal")
    chip(ax, 42, 23.7, "sticky header", "teal", w=14)
    chip(ax, 49, 19.8, "aligned values", "green", w=15)

    box(ax, 69, 19, 26, 31, color="amber", fill="#fffdf6")
    text(ax, 72, 45, "Domain-native templates", size=12.5, weight="bold", color=COLORS["amber"])
    text(ax, 72, 40.5, "Weather, flight, booking, playlist.", size=8.4, color=COLORS["muted"])
    for i, label in enumerate(["daily weather cards", "flight result rows", "booking CTA cards", "playlist track rows"]):
        ax.add_patch(patches.Circle((73, 35.7 - i * 4.7), 0.9, facecolor=COLORS["amber"], edgecolor="none"))
        text(ax, 75, 35.7 - i * 4.7, label, size=8.4)
    save(fig, "fig3_adaptive_table_rendering_v3")


def figure_quality_loop():
    fig, ax = new_fig()
    text(ax, 4, 94, "Validation, Repair, Metrics, and Training Feedback", size=24, weight="bold")
    text(ax, 4, 89.5, "The same contract supports generation-time repair, dataset scoring, and response-to-IR training.", size=12.5, color=COLORS["muted"])

    nodes = [
        (7, 63, "Stage 3 LLM", "candidate IR", "purple", icon_ir),
        (31, 63, "Contract validator", "schema + semantic refs", "green", icon_shield),
        (56, 63, "Native renderer", "Android UI output", "blue", icon_phone),
        (80, 63, "Metrics", "coverage + quality", "amber", None),
    ]
    for x, y, title, sub, color, icon in nodes:
        box(ax, x, y, 16, 15, color=color)
        if icon:
            if title == "Native renderer":
                icon_phone(ax, x + 4.0, y + 7.3, color, scale=0.50)
            else:
                icon(ax, x + 4.0, y + 7.4, color)
            tx = x + 7.3
        else:
            ax.add_patch(patches.Circle((x + 4.0, y + 7.4), 2.7, facecolor=COLORS["amber_soft"], edgecolor=COLORS["amber"], lw=1.2))
            ax.plot([x + 2.8, x + 2.8, x + 4.2, x + 5.7], [y + 5.4, y + 8.2, y + 6.8, y + 9.3], color=COLORS["amber"], lw=2)
            tx = x + 7.3
        text(ax, tx, y + 10.6, title, size=9.7, weight="bold", color=COLORS[color])
        text(ax, tx, y + 6.0, sub, size=7.9, color=COLORS["muted"])
    for a, b in [(23, 31), (47, 56), (72, 80)]:
        arrow(ax, a, 70.5, b, 70.5)

    arrow(ax, 39, 62, 16, 62, color=COLORS["rose"], lw=1.7, rad=.28)
    chip(ax, 20, 55.5, "repair if invalid", "rose", w=18)

    box(ax, 7, 21, 26, 26, color="green", fill="#fbfffb")
    text(ax, 10, 42.5, "Contract diagnostics", size=12.2, weight="bold", color=COLORS["green"])
    text(ax, 10, 38.9, "Non-scoring but actionable.", size=8.4, color=COLORS["muted"])
    for i, item in enumerate(["dangling child refs", "table cell loss risk", "unsupported components", "media/action placement"]):
        ax.add_patch(patches.Circle((10, 35.1 - i * 3.55), 0.65, facecolor=COLORS["green"], edgecolor="none"))
        text(ax, 12, 35.1 - i * 3.55, item, size=8.2)

    box(ax, 38, 21, 25, 26, color="amber", fill="#fffdf6")
    text(ax, 41, 42.5, "Dataset metrics", size=12.2, weight="bold", color=COLORS["amber"])
    text(ax, 41, 38.9, "Score uses semantic IR signals.", size=8.4, color=COLORS["muted"])
    for i, (label, val, color) in enumerate([("schema valid", "100%", "green"), ("content coverage", "↑", "blue"), ("table coverage", "↑", "teal"), ("token size", "↓", "purple")]):
        chip(ax, 41, 33.8 - i * 3.55, f"{label} {val}", color, w=18)

    box(ax, 69, 21, 25, 26, color="purple", fill="#fcfbff")
    text(ax, 72, 42.5, "Training target", size=12.2, weight="bold", color=COLORS["purple"])
    text(ax, 72, 37.0, "Input: Stage 2 response", size=8.8)
    text(ax, 72, 32.8, "Output: strict IR JSON", size=8.8)
    text(ax, 72, 28.6, "Eval: same metrics", size=8.8)
    chip(ax, 72, 23.1, "model-agnostic", "purple", w=17)
    save(fig, "fig4_validation_metrics_training_v3")


def write_critique():
    critique = """# TEDY Paper Figure Critique and v3 Refinement

## Critique pass 1: v2 figure gaps
- The previous figures were structurally correct but too text-heavy for a paper skim.
- Architecture flow lacked enough concrete visual anchors for response, IR, renderer, and mobile UI.
- Table rendering was described as text, not shown as portrait-vs-landscape behavior.
- Validation/metrics were not visual enough to explain why the spec is useful for training and evaluation.

## Refinement pass 1: v3 visual changes
- Added stage icons, semantic chips, mobile mockups, table miniatures, and color-coded lanes.
- Reduced paragraph text inside boxes and moved details into concise labels.
- Added explicit portrait phone, landscape/tablet, and domain-native template panels.
- Added a validation/repair loop and separated diagnostics, metrics, and training target.

## Critique pass 2: paper-readiness checklist
- Readability: high-contrast text on white background, no dense screenshots.
- Reproducibility: deterministic vector generation with PNG, PDF, and SVG outputs.
- Paper fit: 16:9 wide figures suitable for single-column large or two-column page-width placement.
- Visual specificity: includes root/state/elements, compact Table state, native renderer, and mobile adaptation.

## Remaining tradeoff
- These are deterministic diagram figures, not photorealistic AI illustrations. That is intentional for paper clarity and editable vector output.
"""
    (OUT / "visual_critique_v3.md").write_text(critique, encoding="utf-8")


def main():
    figure_architecture()
    figure_spec_anatomy()
    figure_adaptive_rendering()
    figure_quality_loop()
    write_critique()


if __name__ == "__main__":
    main()
