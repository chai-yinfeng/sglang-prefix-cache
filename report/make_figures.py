"""Report figures: the four stress-test scenarios and the three optimizations.

    uv run --no-project --with matplotlib python report/make_figures.py

Reads the archived summaries under experiments/ and writes PNGs to report/pics/, plus PDFs to
report/latex/figures/ when the LaTeX repo (synced with Overleaf) is checked out there.
"""
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "experiments"
OUT = ROOT / "pics"
LATEX_FIGS = ROOT / "latex" / "figures"

# Neutral ink and the first categorical slots of the dataviz reference palette.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
LIGHT_GRAY = "#d4d3cc"
# One color per drafter family; 27B's MTP / DFlash2 take EAGLE3's / DFlash's slot.
FAMILY = {
    "plain": (INK2, "plain"),
    "eagle": (BLUE, "EAGLE3"),
    "dflash": (ORANGE, "DFlash (block 16)"),
    "dspark": (AQUA, "DSpark (block 7)"),
}
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,  # embed TrueType so text in the PDFs stays selectable
})


def rows(path):
    with open(EXP / path) as f:
        return list(csv.DictReader(f))


def family(algo):
    for key, prefixes in (("eagle", ("eagle3", "mtp")), ("dflash", ("dflash",)), ("dspark", ("dspark",))):
        if algo.startswith(prefixes):
            return key
    return "plain"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    if LATEX_FIGS.is_dir():
        fig.savefig(LATEX_FIGS / Path(name).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


def end_labels(ax, items, min_gap, color=INK):
    """Label line ends at (x, y), pushing labels apart vertically so none overlap."""
    items = sorted(items, key=lambda t: t[1])
    placed = [y for _, y, _ in items]
    # Push each crowded pair apart symmetrically, so a cluster stays centered on its points.
    for _ in range(100):
        crowded = False
        for i in range(1, len(placed)):
            gap = placed[i] - placed[i - 1]
            if gap < min_gap - 1e-9:
                placed[i - 1] -= (min_gap - gap) / 2
                placed[i] += (min_gap - gap) / 2
                crowded = True
        if not crowded:
            break
    for (x, y, text), ly in zip(items, placed):
        ax.annotate(text, (x, y), xytext=(x * 1.2, ly), textcoords="data", va="center", fontsize=8.5,
                    color=color)


def concurrency_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("concurrency")


# ---------------------------------------------------------------- introduction diagrams

GOOD, BAD = "#0ca30c", "#d03b3b"


def token(ax, x, y, text, w=0.75, h=0.5, fc="white", ec=AXIS, color=INK, alpha=1.0, fontsize=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec,
                                lw=1.2, alpha=alpha))
    if text:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fontsize, color=color, alpha=alpha,
                fontweight="bold" if bold else "normal")


def diagram_canvas(w, h, title):
    """Axes spanning the whole figure, in inches, so layout is set in data coordinates."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(w / 2, h - 0.25, title, ha="center", va="center", fontsize=13, fontweight="bold")
    return fig, ax


def spec_decode_diagram():
    fig, ax = diagram_canvas(11.5, 4.7, "Speculative decoding: guess cheaply, check in parallel")
    prompt, drafts = ["The", "cat", "sat", "on", "the"], ["mat", "and", "slept", "well"]
    w, step, x0 = 0.8, 0.88, 2.95
    xs_prompt = [x0 + i * step for i in range(len(prompt))]
    xs_draft = [x0 + (len(prompt) + i) * step + 0.15 for i in range(len(drafts))]
    rows = [(3.5, "1. Draft", "a small model guesses\nthe next 4 tokens"),
            (2.15, "2. Verify", "the large target model checks\nall 4 in ONE forward pass"),
            (0.9, "3. Keep", "the accepted prefix plus the\ntarget's own next token")]
    for y, title, sub in rows:
        ax.text(0.1, y + 0.12, title, fontsize=12, fontweight="bold", va="center")
        ax.text(0.1, y - 0.3, sub, fontsize=8.5, color=INK2, va="center")
        for x, t in zip(xs_prompt, prompt):
            token(ax, x, y, t, w=w, fc="#f0efec", alpha=1.0 if y > 3 else 0.45)
    ax.text((xs_prompt[0] + xs_prompt[-1] + w) / 2, 3.93, "context", ha="center", fontsize=8.5, color=MUTED)
    ax.text((xs_draft[0] + xs_draft[-1] + w) / 2, 3.93, "draft guesses", ha="center", fontsize=8.5, color=ORANGE)
    for x, t in zip(xs_draft, drafts):
        token(ax, x, 3.5, t, w=w, fc="#fbe1d6", ec=ORANGE)
    for x, t, ok in zip(xs_draft, drafts, (True, True, False, None)):
        if ok is None:
            token(ax, x, 2.15, t, w=w, fc="white", ec=AXIS, alpha=0.35)
            ax.text(x + w / 2, 2.15 + 0.4, "discarded", ha="center", fontsize=7.5, color=MUTED)
        else:
            token(ax, x, 2.15, t, w=w, fc="white", ec=GOOD if ok else BAD)
            ax.text(x + w - 0.02, 2.15 + 0.36, "✓" if ok else "✗", fontsize=12, fontweight="bold",
                    color=GOOD if ok else BAD, ha="center", va="center")
    ax.text(xs_draft[2] + w / 2, 2.15 - 0.4, "target: purred", ha="center", va="top", fontsize=7.5, color=BAD)
    for x, t, fc, ec in zip(xs_draft[:3], ["mat", "and", "purred"], ["#d7f0d7", "#d7f0d7", "#d6e6fa"],
                            [GOOD, GOOD, BLUE]):
        token(ax, x, 0.9, t, w=w, fc=fc, ec=ec, fontsize=9.5, bold=True)
    ax.text(xs_draft[2] + w / 2, 0.9 - 0.4, "from target", ha="center", va="top", fontsize=7.5, color=BLUE)
    ax.text(xs_draft[3] + 0.05, 0.9, "+3 tokens per\ntarget forward\n(normal decoding: +1)", va="center",
            fontsize=9, fontweight="bold", color=INK)
    for y in (3.5, 2.15):
        ax.annotate("", xy=(1.0, y - 0.9), xytext=(1.0, y - 0.58),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.text(5.75, 0.12, "The output is exactly what the target model alone would produce; speculation only changes "
            "how many target forward passes it takes.", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "intro-1-spec-decode.png")


def drafter_comparison_diagram():
    fig, ax = diagram_canvas(14, 6.9, "Three drafters, three ways to decide how much the target verifies")
    head_y = 6.15
    for x, text in ((0.1, "Drafter"), (2.75, "Draft: how the guesses are made"),
                    (7.25, "Verify: tokens checked per request"), (11.45, "Who sets the verify width")):
        ax.text(x, head_y, text, fontsize=10.5, fontweight="bold", color=INK2, va="center")
    ax.plot([0.1, 13.9], [head_y - 0.3, head_y - 0.3], color=AXIS, lw=1)
    rows = [
        ("EAGLE3", BLUE, "#d6e6fa", 5.0, ["1-layer draft head", "draft KV 4 KiB/token", "accept ~2.3 tokens/step"],
         "3 sequential forwards, one token each", [(4, 2), (4, 3)], "fixed: 4 per request (steps + 1)",
         "Fixed by --speculative-num-steps.\nSGLang adaptive can switch the\nstep count by batch size and\n"
         "acceptance, but every choice\nneeds its own CUDA graphs."),
        ("DFlash", ORANGE, "#fbe1d6", 3.15, ["5-layer block drafter", "draft KV 20 KiB/token", "accept ~2.2 tokens/step"],
         "1 forward drafts the whole block in parallel", [(16, 2), (16, 3)],
         "fixed: 16 per request, most rejected on chat", "Fixed by the block size (16).\nNo runtime knob in SGLang."),
        ("DSpark", AQUA, "#d3f0e5", 1.3, ["5-layer block drafter", "+ confidence head", "draft KV 20 KiB/token",
                                           "accept ~2.8 tokens/step"],
         "1 forward drafts 7 tokens, each with a confidence", [(7, 4), (2, 1)],
         "differs per request (up to 8)", "Per request, every step: the\nSGLang planner weighs each\ntoken's "
         "confidence against\nthe measured cost of a step."),
    ]
    for name, color, light, y, facts, draft_note, verify, verify_note, policy in rows:
        ax.text(0.1, y + 0.55, name, fontsize=14, fontweight="bold", color=color, va="center")
        for i, f in enumerate(facts):
            ax.text(0.1, y + 0.12 - i * 0.3, f, fontsize=8.5, color=INK2, va="center")
        if name == "EAGLE3":
            for i in range(3):
                x = 3.0 + i * 1.3
                token(ax, x, y + 0.3, f"t{i + 1}", w=0.6, h=0.45, fc=light, ec=color)
                ax.text(x + 0.3, y - 0.12, f"fwd {i + 1}", ha="center", fontsize=8, color=INK2)
                if i < 2:
                    ax.annotate("", xy=(x + 1.27, y + 0.3), xytext=(x + 0.63, y + 0.3),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
        else:
            n, w, gap = (15, 0.24, 0.03) if name == "DFlash" else (7, 0.5, 0.08)
            for i in range(n):
                token(ax, 2.85 + i * (w + gap), y + 0.3, "", w=w, h=0.45, fc=light, ec=color)
            right = 2.85 + n * (w + gap) - gap
            if name == "DSpark":
                for i, c in enumerate([0.95, 0.85, 0.7, 0.5, 0.35, 0.25, 0.15]):
                    ax.add_patch(Rectangle((2.85 + i * (w + gap) + 0.12, y + 0.56), w - 0.24, 0.28 * c,
                                           fc=color, ec="none"))
                ax.text(right + 0.1, y + 0.65, "confidence", fontsize=7.5, color=INK2, va="center")
            ax.plot([2.85, 2.85, right, right], [y - 0.02, y - 0.08, y - 0.08, y - 0.02], color=INK2, lw=1)
            ax.text((2.85 + right) / 2, y - 0.2, "1 forward", ha="center", va="top", fontsize=8, color=INK2)
        ax.text(2.85, y - 0.55, draft_note, fontsize=8.5, color=INK, va="center")
        for r, (n, ok) in enumerate(verify):
            ry = y + 0.45 - r * 0.5
            ax.text(7.25, ry, "AB"[r], fontsize=8.5, color=INK2, va="center", fontweight="bold")
            for i in range(n):
                accepted = i < ok
                token(ax, 7.55 + i * 0.235, ry, "", w=0.205, h=0.36, fc=color if accepted else "#ecebe6",
                      ec=color if accepted else AXIS)
        ax.text(7.25, y - 0.55, verify_note, fontsize=8.5, color=INK, va="center")
        ax.text(11.45, y + 0.2, policy, fontsize=8.5, color=INK, va="center", linespacing=1.35)
        if y > 2:
            ax.plot([0.1, 13.9], [y - 0.95, y - 0.95], color=GRID, lw=1)
    legend = [Patch(fc="#9a9993", ec="none"), Patch(fc="#ecebe6", ec=AXIS)]
    ax.legend(legend, ["accepted (filled in the drafter's color)", "checked by the target, then rejected (wasted)"],
              loc="lower center", bbox_to_anchor=(0.55, 0.045), ncol=2, fontsize=8.5)
    ax.text(7, 0.12, "Configurations as served in this report (EAGLE3 steps 3, DFlash block 16, DSpark block 7). "
            "Acceptance: Qwen3-8B, natural chat. Verify rows A / B are illustrations.", ha="center", fontsize=8.5,
            color=MUTED)
    save(fig, "intro-2-drafters.png")


# ---------------------------------------------------------------- stress-test data

MODELS = ["Qwen3-4B", "Qwen3-8B", "Qwen3.8-27B"]
MODEL_TITLE = {
    "Qwen3-4B": "Qwen3-4B (dense)",
    "Qwen3-8B": "Qwen3-8B (dense)",
    "Qwen3.8-27B": "Qwen3.8-27B NVFP4 (hybrid)",
}
SPEC_27B = {"mtp-n4-nopg": "eagle", "dflash2-n3-nopg": "dflash", "dspark-n2-nopg": "dspark"}
LABEL_27B = {"eagle": "MTP", "dflash": "DFlash2", "dspark": "DSpark"}


def speedups(workload):
    """{model: {family: [(concurrency, x plain), ...]}} for one workload."""
    out = {m: {} for m in MODELS}

    def add(model, algo, conc, value):
        fam = family(algo)
        if fam != "plain":
            out[model].setdefault(fam, []).append((int(conc), float(value)))

    if workload in ("w1-natural", "w2-shared-prefix"):
        for r in rows("stress-gapfill-v0.5.22/gf-4b-e2.csv"):
            if r["workload"] == workload and r["algorithm"] in ("eagle3", "dflash", "dspark"):
                add("Qwen3-4B", r["algorithm"], r["concurrency"], r["vs_plain"])
        for r in rows("stress-qwen3-8b-v0.5.22/e2-v1/summary.csv"):
            if r["workload"] == workload and r["algorithm"] in ("eagle3", "dflash", "dspark"):
                add("Qwen3-8B", r["algorithm"], r["concurrency"], r["speedup_vs_plain"])
        for r in rows("stress-qwen38-27b-v0.5.22/e2-v1/summary.csv"):
            if r["workload"] == workload and r["algorithm"] in SPEC_27B:
                add("Qwen3.8-27B", r["algorithm"], r["concurrency"], r["speedup_vs_plain"])
    else:
        name = {"multiturn": "gf-multiturn.csv", "longgen": "gf-longgen.csv"}[workload]
        for r in rows(f"stress-gapfill-v0.5.22/{name}"):
            if r["algorithm"] != "plain":
                add(r["model"], r["algorithm"], r["concurrency"], r["vs_plain"])
    for model in out.values():
        for series in model.values():
            series.sort()
    return out


def speedup_panels(workload, title, name, note):
    data = speedups(workload)
    y_top = max(y for m in data.values() for s in m.values() for _, y in s) * 1.12
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, model in zip(axes, MODELS):
        ax.axhspan(0, 1, color="#f6e3e2", zorder=0, lw=0)
        ax.axhline(1, color=INK2, lw=1.2, ls="--", zorder=1)
        ticks = sorted({c for s in data[model].values() for c, _ in s})
        ends = []
        for fam in ("eagle", "dflash", "dspark"):
            series = data[model].get(fam)
            if not series:
                continue
            xs, ys = zip(*series)
            label = LABEL_27B[fam] if model == "Qwen3.8-27B" else FAMILY[fam][1].split(" ")[0]
            ax.plot(xs, ys, color=FAMILY[fam][0], lw=2, marker="o", ms=5, zorder=3)
            ends.append((xs[-1], ys[-1], f"{label} {ys[-1]:.2f}x"))
        end_labels(ax, ends, min_gap=0.075 * y_top)
        concurrency_axis(ax, ticks)
        ax.set_xlim(ticks[0] / 1.3, ticks[-1] * 3.4)
        ax.set_title(MODEL_TITLE[model])
        ax.text(0.03, 0.04, "slower than plain", transform=ax.transAxes, fontsize=8, color="#b0403f")
    axes[0].set_ylabel("throughput relative to plain (x)")
    axes[0].set_ylim(0, y_top)
    handles = [Line2D([], [], color=FAMILY[f][0], lw=2, marker="o", ms=5) for f in ("eagle", "dflash", "dspark")]
    labels = ["EAGLE3 (27B: MTP)", "DFlash (27B: DFlash2)", "DSpark"]
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.09, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.1, note, ha="center", fontsize=8.5, color=MUTED)
    save(fig, name)


def overview_heatmap():
    """x plain at concurrency 1 and at each model's highest concurrency, every scenario."""
    workloads = [("w1-natural", "natural chat\n(reference)"), ("w2-shared-prefix", "shared\nprefix"),
                 ("multiturn", "multi-turn\nchat"), ("longgen", "short in,\nlong out")]
    data = {w: speedups(w) for w, _ in workloads}
    row_keys = [(m, f) for m in MODELS for f in ("eagle", "dflash", "dspark")]
    row_labels = [f"{m.replace('Qwen3.8-', '').replace('Qwen3-', '')}  "
                  f"{LABEL_27B[f] if m == 'Qwen3.8-27B' else FAMILY[f][1].split(' ')[0]}" for m, f in row_keys]
    cmap = LinearSegmentedColormap.from_list("div", ["#c0302f", "#e88b8a", "#f0efec", "#86b6ef", "#1c5cab"])
    norm = TwoSlopeNorm(vmin=math.log2(0.25), vcenter=0, vmax=math.log2(4))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True)
    for ax, which, title in ((axes[0], 0, "low load\n(concurrency 1)"),
                             (axes[1], -1, "high load\n(concurrency 64; 27B: 16)")):
        grid = [[data[w][m][f][which][1] for w, _ in workloads] for m, f in row_keys]
        ax.imshow([[math.log2(v) for v in r] for r in grid], cmap=cmap, norm=norm, aspect="auto")
        for i, r in enumerate(grid):
            for j, v in enumerate(r):
                strong = abs(math.log2(v)) > 1.0
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if strong else INK, fontweight="bold" if v < 1 else "normal")
        ax.set_xticks(range(len(workloads)))
        ax.set_xticklabels([lab for _, lab in workloads], fontsize=8.5)
        ax.set_yticks(range(len(row_keys)))
        ax.set_yticklabels(row_labels)
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        for y in (2.5, 5.5):
            ax.axhline(y, color="white", lw=3)
        ax.set_title(title)
    fig.suptitle("Speculative decoding vs plain: wins at low load, loses at high load",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02, "Cells: output throughput relative to plain (x). Blue = faster than plain, red = slower. "
             "27B spec configs use hand-tuned pinned pools (E1).", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "stress-0-overview.png")


def multiturn_history():
    data = {(r["model"], r["algorithm"]): r for r in rows("stress-gapfill-v0.5.22/gf-multiturn.csv")
            if r["concurrency"] == "64"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    algos = ["plain", "eagle3", "dflash", "dspark"]
    width = 0.2
    for ax, model in zip(axes, ("Qwen3-4B", "Qwen3-8B")):
        for k, algo in enumerate(algos):
            r = data[(model, algo)]
            ys = [float(r[f"hit_round{i}"]) for i in (2, 3, 4)]
            xs = [i + (k - 1.5) * width for i in range(3)]
            ax.bar(xs, ys, width * 0.9, color=FAMILY[family(algo)][0], label=FAMILY[family(algo)][1].split(" ")[0])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["round 2", "round 3", "round 4"])
        kv = {a: int(data[(model, a)]["kv_tokens"]) for a in algos}
        spec_kv = [v for a, v in kv.items() if a != "plain"]
        ax.set_title(f"{model}, concurrency 64\nKV pool: plain {kv['plain'] / 1000:.0f}K tokens, "
                     f"spec {min(spec_kv) / 1000:.0f}K-{max(spec_kv) / 1000:.0f}K", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("prefix hit rate of the round")
    axes[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(1.1, 1.34))
    fig.suptitle("Multi-turn chat: does the conversation history survive until the next round?",
                 y=1.2, fontsize=12, fontweight="bold")
    save(fig, "stress-2b-multiturn-history.png")


def memory_breakdown():
    files = {"Qwen3-4B": "memory-timeline-4b-w2c64-startup.csv", "Qwen3-8B": "memory-timeline-8b-w2c64-startup.csv",
             "Qwen3.8-27B": "memory-timeline-27b-startup.csv"}
    parts = [("target weights", BLUE), ("draft weights", ORANGE), ("KV pool (27B: KV + state)", AQUA),
             ("draft KV", YELLOW), ("CUDA graphs", MAGENTA)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, model in zip(axes, MODELS):
        configs, stacks, pool_labels, freed = [], [], [], {}
        by_config = {}
        for r in rows(f"stress-gapfill-v0.5.22/{files[model]}"):
            by_config.setdefault(r["config"], []).append(r)
        for config, events in by_config.items():
            seg = dict.fromkeys(range(len(parts)), 0.0)
            weights = 0
            for e in events:
                if not e["delta_gb"]:
                    continue
                delta, ev = float(e["delta_gb"]), e["event"]
                if ev.startswith("weights"):
                    seg[0 if weights == 0 else 1] += delta
                    weights += 1
                elif ev.startswith("draft KV"):
                    seg[3] += delta
                elif "KV pool" in ev:
                    seg[2] += delta
                    tokens = int(re.search(r"KV pool (\d+) tokens", ev).group(1))
                    slots = re.search(r"state pool (\d+) slots", ev)
                    pool_labels.append(f"{tokens / 1000:.1f}K tok" + (f"\n{slots.group(1)} slots" if slots else ""))
                elif ev.startswith("graph") and delta < 0:
                    # MTP: the head's own embedding / lm_head is released while graphs are captured,
                    # after the pools were sized from the larger footprint (stress test problem 7).
                    freed[len(configs)] = freed.get(len(configs), 0.0) - delta
                elif ev.startswith("graph"):
                    seg[4] += delta
            configs.append({"plain": "plain", "eagle3": "EAGLE3", "dflash": "DFlash", "dspark": "DSpark",
                            "mtp": "MTP", "dflash2": "DFlash2"}[config])
            stacks.append(seg)
        bottoms = [0.0] * len(configs)
        for p, (label, color) in enumerate(parts):
            vals = [s[p] for s in stacks]
            ax.bar(range(len(configs)), vals, 0.62, bottom=bottoms, color=color, edgecolor="white", lw=1.2,
                   label=label)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        for i, gb in freed.items():
            ax.annotate(f"{gb:.1f} GB freed later,\nsits idle", (i, bottoms[i]), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        for i, b in enumerate(bottoms):
            if i not in freed:
                ax.text(i, b + 0.3, f"{b:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([f"{c}\n{p}" for c, p in zip(configs, pool_labels)], fontsize=8.5)
        ax.set_title(MODEL_TITLE[model])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 34)
    axes[0].set_ylabel("GPU memory used at startup (GB)")
    handles = [Patch(color=c) for _, c in parts]
    fig.legend(handles, [lab for lab, _ in parts], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Where the memory goes: draft weights and graphs come out of the target's KV pool",
                 y=1.1, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.08, "From server-log 'avail mem' after each startup stage. Under each bar: target KV capacity "
             "(27B: and state slots). 27B spec configs use hand-tuned pinned pools.", ha="center", fontsize=8.5,
             color=MUTED)
    save(fig, "stress-3a-memory-breakdown.png")


def same_capacity():
    panels = []
    r4 = rows("stress-gapfill-v0.5.22/gf-4b-e2.csv")
    pick = lambda rs, algo, key="output_tok_s", wl="w2-shared-prefix", col="algorithm": sorted(
        (int(r["concurrency"]), float(r[key])) for r in rs if r.get("workload", wl) == wl and r[col] == algo)
    panels.append(("Qwen3-4B", pick(r4, "plain"), pick(r4, "plain-kv93439"), pick(r4, "dspark"), "94K"))
    r8 = rows("stress-qwen3-8b-v0.5.22/e2-v1/summary.csv")
    panels.append(("Qwen3-8B", pick(r8, "plain"), pick(r8, "plain-kv38093"), pick(r8, "dspark"), "38K"))
    r27 = rows("phase1-followups-v0.5.22/q3q4-27b-serving.csv")
    panels.append(("Qwen3.8-27B", pick(r27, "plain", col="config"), pick(r27, "plain-eq-dspark-n2", col="config"),
                   pick(r27, "dspark-n2-nopg", col="config"), "DSpark's"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (model, full, capped, spec, cap) in zip(axes, panels):
        series = [(full, INK2, "-", "plain, full pool"), (capped, INK2, "--", "plain, pool cut to DSpark's"),
                  (spec, AQUA, "-", "DSpark")]
        for data, color, ls, _ in series:
            xs, ys = zip(*data)
            ax.plot(xs, ys, color=color, ls=ls, lw=2, marker="o", ms=4.5, mfc="white" if ls == "--" else color)
        x_end = full[-1][0]
        f, c, s = full[-1][1], capped[-1][1], spec[-1][1]
        y_top = max(f, c, s) * 1.15
        ax.set_ylim(0, y_top)
        end_labels(ax, [(x_end, f, "plain"), (x_end, c, f"cut plain\n{(c / f - 1) * 100:+.0f}%"),
                        (x_end, s, f"DSpark\n{(s / c - 1) * 100:+.0f}% vs cut")], min_gap=0.13 * y_top)
        concurrency_axis(ax, [x for x, _ in full])
        ax.set_xlim(full[0][0] / 1.3, x_end * 3.6)
        ax.set_title(MODEL_TITLE[model])
    axes[0].set_ylabel("output throughput (tok/s)")
    handles = [Line2D([], [], color=c, ls=ls, lw=2, marker="o", ms=4.5, mfc="white" if ls == "--" else c)
               for _, c, ls, _ in [(None, INK2, "-", 0), (None, INK2, "--", 0), (None, AQUA, "-", 0)]]
    fig.legend(handles, ["plain, full pool", "plain, pool cut to DSpark's size", "DSpark"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Limited memory: the loss is the pool size, not speculation (shared-prefix workload)",
                 y=1.09, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.12, "8B: cutting plain to DSpark's KV costs 46%, and DSpark beats plain at equal capacity. "
             "4B: the pool still holds the hot prefixes, so cutting costs nothing\n(4B points at concurrency "
             "32-64 vary about 20% between plain runs). 27B: plain gets DSpark's exact KV + state pool (2 requests).",
             ha="center", fontsize=8.5, color=MUTED)
    save(fig, "stress-3b-same-capacity.png")


def serving_timeline():
    by = {}
    for r in rows("stress-gapfill-v0.5.22/memory-timeline-8b-w2c64-serving.csv"):
        by.setdefault(r["config"], []).append(r)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True, gridspec_kw={"hspace": 0.15})
    for config in ("plain", "eagle3", "dflash", "dspark"):
        rs = by[config]
        t = [float(r["t_s"]) for r in rs]
        used = [100 * float(r["kv_used_tokens"]) / float(r["kv_total_tokens"]) for r in rs]
        running = [float(r["running"]) for r in rs]
        active = [x for x in running if x > 0]
        color, label = FAMILY[family(config)]
        label = label.split(" ")[0]
        kv = float(rs[0]["kv_total_tokens"])
        top.plot(t, used, color=color, lw=1.8, label=f"{label} ({kv / 1000:.1f}K tok)")
        bottom.plot(t, running, color=color, lw=1.8, label=f"{label}: {sum(active) / len(active):.1f} on average")
    top.set_ylabel("KV pool used (%)")
    top.set_ylim(0, 105)
    top.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, title="KV capacity",
               title_fontsize=8.5)
    bottom.set_ylabel("running requests")
    bottom.set_xlabel("time since the benchmark started (s)")
    bottom.axhline(64, color=MUTED, lw=1, ls=":")
    bottom.text(1, 60, "64 clients", color=MUTED, fontsize=8)
    bottom.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, title="running requests",
                  title_fontsize=8.5)
    bottom.set_ylim(0, 70)
    fig.suptitle("Qwen3-8B, shared prefix, concurrency 64: every pool fills up;\n"
                 "the spec configs fit fewer requests into it", fontsize=12, fontweight="bold", y=1.0)
    save(fig, "stress-3c-serving-timeline.png")


# ---------------------------------------------------------------- optimization figures

def dflash_width_by_load():
    data = rows("policy-round4-v0.5.22/w1-static-width.csv")
    tput = {(r["algo"], r["workload"], int(r["concurrency"])): float(r["output_tok_s"]) for r in data}
    width_color = dict(zip([2, 3, 4, 8, 16], BLUE_RAMP))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)
    for ax, (wl, title, widths) in zip(axes, (("w1-natural", "natural chat (acceptance ~2)", [2, 3, 4, 8, 16]),
                                              ("w-edit", "proofreading (acceptance ~12)", [2, 4, 8, 16]))):
        concs = [1, 4, 16, 64]
        best = {}
        for w in widths:
            ys = [tput[(f"dflash-pcs-b{w}", wl, c)] / tput[("plain-pcs-default", wl, c)] for c in concs]
            for c, y in zip(concs, ys):
                if y > best.get(c, (0, 0))[1]:
                    best[c] = (w, y)
            ax.plot(concs, ys, color=width_color[w], lw=2, marker="o", ms=5)
        for c, (w, y) in best.items():
            ax.annotate(f"best: {w}", (c, y), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8.5, fontweight="bold", color=INK)
        ax.axhline(1, color=INK2, lw=1.2, ls="--")
        concurrency_axis(ax, concs)
        ax.set_xlim(0.6, 110)
        ax.set_title(title)
        ax.set_ylim(0, max(y for _, y in best.values()) * 1.2)
    axes[0].set_ylabel("throughput relative to plain (x)")
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=5) for c in width_color.values()]
    fig.legend(handles, [f"width {w}" for w in width_color], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.13), title="DFlash verify width (tokens per request per step)",
               title_fontsize=9)
    fig.suptitle("DFlash on Qwen3-8B: the best static verify width depends on load and workload",
                 y=1.2, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.1, "Natural chat: wide verify wins at low load, narrow at concurrency 64 (most of the 16 "
             "verified tokens are rejected).\nProofreading accepts ~12 tokens per step, so wide verify stays "
             "best up to concurrency 16.", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "opt-1a-dflash-width-by-load.png")


def verify_width_results():
    phase = rows("opt-align-v0.5.22/o1-phase-benches.csv")
    walls = {(r["bench"], r["algorithm"]): float(r["total_wall_s"]) for r in phase}
    for r in rows("opt-align-v0.5.22/o1-repeats.csv"):
        walls[(r["bench"], r["algorithm"])] = float(r["mean_wall_s"])
    benches = [("natural", "chat\nc4 <-> c64"), ("edit", "proofreading\nc4 <-> c64"), ("mixed", "mixed\n6 phases")]
    panels = [
        ("DFlash", "plain-pcs-default", "dflash-pcs-b16", ["dflash-pcs-b16", "dflash-pcs-b8", "dflash-pcs-b4",
                                                          "dflash-pcs-b3"], "dflash-pcs-width-g",
         "dynamic: our width policy"),
        ("EAGLE3", "plain-pcs-default", "eagle3-pcs-s3", ["eagle3-pcs-s1", "eagle3-pcs-s2", "eagle3-pcs-s3",
                                                         "eagle3-pcs-s5"], "eagle3-pcs-width-s123",
         "dynamic: our width policy (steps 1/2/3)"),
        ("DSpark", "plain-fa4-pcs", "dspark-pcs", [], "dspark-fa4-planner",
         "dynamic: SGLang confidence planner"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    w = 0.26
    for ax, (name, plain, default, statics, dynamic, dyn_label) in zip(axes, panels):
        for i, (bench, _) in enumerate(benches):
            base = walls[(bench, plain)]
            cand = [(walls[(bench, a)], a) for a in statics if (bench, a) in walls]
            best_wall, best_algo = min(cand) if cand else (None, None)
            bars = [(base / walls[(bench, default)], LIGHT_GRAY, None)]
            if best_wall is not None:
                tag = best_algo.rsplit("-", 1)[1]
                bars.append((base / best_wall, MUTED, tag))
            bars.append((base / walls[(bench, dynamic)], BLUE, None))
            offsets = [-w, 0, w] if len(bars) == 3 else [-w / 2, w / 2]
            for (y, color, tag), dx in zip(bars, offsets):
                ax.bar(i + dx, y, w * 0.9, color=color)
                if tag:
                    ax.text(i + dx, y - 0.05, tag, ha="center", va="top", fontsize=7.5, color="white",
                            fontweight="bold")
            ref = bars[-2][0]
            gain = (bars[-1][0] / ref - 1) * 100
            ax.text(i + offsets[-1], bars[-1][0] + 0.03, f"{gain:+.0f}%", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=INK)
        ax.axhline(1, color=INK2, lw=1.2, ls="--")
        ax.set_xticks(range(len(benches)))
        ax.set_xticklabels([lab for _, lab in benches], fontsize=8.5)
        ax.set_title(f"{name}\n{dyn_label}", fontsize=10.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("speedup over plain (total time, x)")
    axes[0].set_ylim(0, 2.95)
    handles = [Patch(color=LIGHT_GRAY), Patch(color=MUTED), Patch(color=BLUE)]
    fig.legend(handles, ["default static width", "best static width (label = which)", "dynamic width"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Dynamic verify width on Qwen3-8B: % on the blue bar = gain over the best static width",
                 y=1.2, fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.1, "Time-varying phase benches (load and workload switch mid-run). DSpark's only static "
             "config is full verify (block 7); our load-aware policy ties the planner.\n"
             "EAGLE3 / DSpark bars: mean of 3 passes; DFlash: rounds 4-5.", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "opt-1b-verify-width-results.png")


def hybrid_split():
    # Qwen3.5-4B, w1-natural c64 (opt-align FINDINGS O2 table): (label, D, default tok/s, cap, policy tok/s, cap).
    # EAGLE3 / DSpark from o2-hybrid-split.csv, MTP / DFlash from round 5 m1, defaults from round 4.
    four_b = [("MTP", 4, 1741, 12, 1998, 16), ("EAGLE3", 4, 1642, 12, 1881, 16), ("DFlash b4", 4, 1918, 12, 2245, 16),
              ("DFlash b16", 16, 1000, 4, 1364, 8), ("DSpark b16", 17, 499, 4, 716, 8)]
    plain_c64 = 1916
    # Qwen3.8-27B: max running requests per split; 0 = fails to start (o2-hybrid-split.csv, stress E2 pinned pools).
    big = [("DSpark", [0, 1, 1, 2]), ("DFlash2", [0, 1, 2, 3]), ("MTP", [0, 1, 1, 4])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
    w = 0.38
    for i, (name, d, base, cap0, pol, cap1) in enumerate(four_b):
        a.bar(i - w / 2, base, w * 0.92, color=LIGHT_GRAY)
        a.bar(i + w / 2, pol, w * 0.92, color=BLUE)
        a.text(i + w / 2, pol + 30, f"+{(pol / base - 1) * 100:.0f}%", ha="center", va="bottom", fontsize=9,
               fontweight="bold")
        a.text(i - w / 2, base / 2, f"{cap0}", ha="center", va="center", fontsize=8, color=INK2)
        a.text(i + w / 2, pol / 2, f"{cap1}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    a.axhline(plain_c64, color=INK2, lw=1.2, ls="--")
    a.text(4.45, plain_c64 + 30, "plain", ha="right", va="bottom", fontsize=8.5, color=INK2)
    a.set_xticks(range(len(four_b)))
    a.set_xticklabels([f"{n}\nD = {d}" for n, d, *_ in four_b], fontsize=9)
    a.set_ylabel("output throughput (tok/s)")
    a.set_ylim(0, 2600)
    a.grid(axis="x", visible=False)
    a.set_title("Qwen3.5-4B, natural chat, concurrency 64\n(number in the bar = request cap)", fontsize=10.5)
    a.legend([Patch(color=LIGHT_GRAY), Patch(color=BLUE)], ["default split (0.9)", "memory policy: x (S + D) / S"],
             loc="upper right", fontsize=8.5)
    shades = [LIGHT_GRAY, BLUE, MUTED, INK2]
    names = ["default split (0.9)", "memory policy", "cookbook ratio", "hand-tuned pinned pools"]
    w2 = 0.2
    for i, (name, caps) in enumerate(big):
        for k, (cap, color) in enumerate(zip(caps, shades)):
            x = i + (k - 1.5) * w2
            if cap == 0:
                b.plot(x, 0.12, marker="x", ms=8, mew=2, color="#c0302f")
            else:
                b.bar(x, cap, w2 * 0.9, color=color)
                b.text(x, cap + 0.08, str(cap), ha="center", va="bottom", fontsize=8.5, color=INK)
    b.set_xticks(range(len(big)))
    b.set_xticklabels([n for n, _ in big])
    b.set_ylabel("max running requests")
    b.set_ylim(0, 5)
    b.grid(axis="x", visible=False)
    b.set_title("Qwen3.8-27B NVFP4 (plain: 15 requests)\nx = fails to start", fontsize=10.5)
    b.legend([Patch(color=c) for c in shades], names, loc="upper left", fontsize=8)
    fig.suptitle("Hybrid model: give the state pool room for the verify states (D per request)",
                 y=1.05, fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.07, "4B: the gain tracks D (D = 4: +15-17%, D = 16-17: +36-43%). 27B: the policy makes every "
             "drafter start, but total memory still limits it to 1 request.\n27B MTP: default and policy run without "
             "ReplaySSM (D = 4); cookbook and pinned pools use ReplaySSM (D = 0). Pinned pools also drop the target's "
             "prefill CUDA graph.", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "opt-2-hybrid-split.png")


def hicache():
    data = {(r["workload"], r["drafter"], r["hicache"]): r for r in rows("opt-align-v0.5.22/o3-hicache.csv")
            if r["concurrency"] == "64"}
    drafters = ["plain", "EAGLE3", "DFlash", "DSpark"]
    kv = {d: int(data[("w2-shared-prefix", d, "off")]["kv_tokens"]) for d in drafters}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    w = 0.38
    for col, (wl, title) in enumerate((("w2-shared-prefix", "shared prefix (W2)"), ("w-multiturn", "multi-turn chat"))):
        top, bottom = axes[0][col], axes[1][col]
        plain_off = float(data[(wl, "plain", "off")]["output_tok_s"])
        for i, d in enumerate(drafters):
            off, on = data[(wl, d, "off")], data[(wl, d, "on")]
            t_off, t_on = float(off["output_tok_s"]), float(on["output_tok_s"])
            top.bar(i - w / 2, t_off, w * 0.92, color=LIGHT_GRAY)
            top.bar(i + w / 2, t_on, w * 0.92, color=BLUE)
            top.text(i + w / 2, t_on + 40, f"{(t_on / t_off - 1) * 100:+.0f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
            h_off, h_on = float(off["hit_rate"]), float(on["hit_rate"])
            bottom.bar(i - w / 2, h_off, w * 0.92, color=LIGHT_GRAY)
            bottom.bar(i + w / 2, h_on, w * 0.92, color=BLUE)
            bottom.text(i - w / 2, h_off + 0.02, f"{h_off:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
            bottom.text(i + w / 2, h_on + 0.02, f"{h_on:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        top.axhline(plain_off, color=INK2, lw=1.2, ls="--")
        top.set_title(title)
        top.set_ylim(0, 3900)
        bottom.set_ylim(0, 1.15)
        for ax in (top, bottom):
            ax.grid(axis="x", visible=False)
        bottom.set_xticks(range(len(drafters)))
        bottom.set_xticklabels([f"{d}\nKV {kv[d] / 1000:.0f}K" for d in drafters], fontsize=9)
    axes[0][0].set_ylabel("output throughput (tok/s)")
    axes[1][0].set_ylabel("prefix hit rate")
    fig.legend([Patch(color=LIGHT_GRAY), Patch(color=BLUE), Line2D([], [], color=INK2, ls="--")],
               ["HiCache off", "HiCache on (12 GB host pool)", "plain without HiCache"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("HiCache on Qwen3-8B, concurrency 64: the more KV speculation took, the more HiCache gives back",
                 y=1.07, fontsize=12, fontweight="bold")
    save(fig, "opt-3-hicache.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    spec_decode_diagram()
    drafter_comparison_diagram()
    overview_heatmap()
    speedup_panels("w2-shared-prefix", "Scenario 1 - shared prefix (W2): repeated 4K-token document prefixes",
                   "stress-1-shared-prefix.png", "24 document prefixes x 4K tokens, Zipf access, 128 output tokens. "
                   "27B spec configs use hand-tuned pinned pools (E1).")
    speedup_panels("multiturn", "Scenario 2 - multi-turn chat: each round appends to the full history",
                   "stress-2-multiturn.png", "4 rounds per conversation; the next prompt is the full previous "
                   "token sequence plus a new user turn. Up to 512 output tokens per round.")
    multiturn_history()
    memory_breakdown()
    same_capacity()
    serving_timeline()
    speedup_panels("longgen", "Scenario 4 - short input, long output (thinking mode)",
                   "stress-4-long-output.png", "~85 input tokens, ~1,600-1,800 output tokens per request; "
                   "the KV pool fills up through decode, not through prefixes.")
    dflash_width_by_load()
    verify_width_results()
    hybrid_split()
    hicache()
