"""
Generate the data-derived figures for the Arogya AI paper.

Every number plotted here comes from a table already in the paper:
  fig_differential  <- Table V   (top-ranked conditions)
  fig_latency       <- Table III (latency before/after the two optimisations)
  fig_trace         <- Table VI  (per-turn processing trace)

Nothing is invented. Re-run with:  python make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# --- palette -------------------------------------------------------------
# Emphasis hue and de-emphasis gray chosen for greyscale print:
#   #184f95  contrast 8.10:1 vs white, greyscale L 31.6%
#   #8c8c8c  contrast 3.36:1 vs white, greyscale L 54.4%   -> 22.8 pts apart
ACCENT = "#184f95"   # emphasised series
PLAIN = "#2a78d6"   # single-series fill
CONTEXT = "#8c8c8c"   # de-emphasised context
INK = "#0b0b0b"   # primary text
INK2 = "#52514e"   # secondary text
MUTED = "#898781"   # axis labels
GRID = "#e1e0d9"   # hairline gridline
BASE = "#c3c2b7"   # baseline / axis rule

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": BASE,
    "xtick.color": MUTED,
    "ytick.color": INK2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def tidy(ax, xgrid=True):
    """Hairline, recessive chrome: solid grid, no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    if xgrid:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5, linestyle="-")
        ax.yaxis.grid(False)
    ax.tick_params(length=2, width=0.6)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


# =========================================================================
# Figure: confidence-ranked differential  (Table V)
# Form = emphasis: one condition dominates, the rest are context.
# =========================================================================
def fig_differential():
    labels = ["Malaria", "Common cold", "Dengue", "Typhoid", "Other (aggregate)"]
    values = [83.6, 3.9, 3.6, 3.4, 5.5]
    colors = [ACCENT] + [CONTEXT] * 4

    fig, ax = plt.subplots(figsize=(3.4, 1.65))
    y = range(len(labels))
    ax.barh(y, values, height=0.6, color=colors, linewidth=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Posterior probability (%)")
    tidy(ax)

    for i, v in enumerate(values):
        ax.text(v + 1.6, i, f"{v}", va="center", ha="left",
                fontsize=6.5, color=INK if i == 0 else INK2)
    save(fig, "fig_differential")


# =========================================================================
# Figure: latency before / after  (Table III)
# Form = dumbbell (before -> after per item), one hue.
# Hollow vs filled marker carries the distinction in greyscale.
# =========================================================================
def fig_latency():
    rows = [
        ("Startup (until serving)", 27000, 1200, "27 s", "1.2 s"),
        ("Per turn",                  960,   82, "960 ms", "82 ms"),
        ("Symptom extraction",        891,   27, "891 ms", "27 ms"),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    for i, (_, before, after, lb, la) in enumerate(rows):
        ax.add_patch(FancyArrowPatch(
            (before, i), (after, i),
            arrowstyle="-|>", mutation_scale=6,
            linewidth=1.0, color=BASE, shrinkA=3.5, shrinkB=3.5,
            zorder=1))
        ax.plot(before, i, "o", markersize=5, markerfacecolor="white",
                markeredgecolor=INK2, markeredgewidth=1.0, zorder=2)
        ax.plot(after, i, "o", markersize=5, markerfacecolor=ACCENT,
                markeredgecolor=ACCENT, zorder=2)
        ax.text(before * 1.25, i, lb, va="center", ha="left",
                fontsize=6, color=INK2)
        # "after" label sits above its marker: placing it to the left runs it
        # into the y-axis tick labels on the fastest row.
        ax.text(after, i - 0.22, la, va="bottom", ha="center",
                fontsize=6, color=ACCENT, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(8, 120000)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms, logarithmic scale)")
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["10", "100", "1k", "10k", "100k"])
    tidy(ax)

    h_before = plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                          markerfacecolor="white", markeredgecolor=INK2,
                          markeredgewidth=1.0, label="before")
    h_after = plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                         markerfacecolor=ACCENT, markeredgecolor=ACCENT,
                         label="after")
    ax.legend(handles=[h_before, h_after], loc="upper left",
              frameon=False, handletextpad=0.4, borderaxespad=0.0,
              labelcolor=INK2, ncol=2, columnspacing=1.0)
    save(fig, "fig_latency")


# =========================================================================
# Figure: per-turn processing trace  (Table VI)
# Form = single series, one colour for every bar (stages are nominal).
# =========================================================================
def fig_trace():
    labels = ["Symptom extraction", "Red-flag safety scan",
              "Naive Bayes prediction", "Follow-up selection"]
    values = [27.0, 2.4, 45.3, 7.5]

    fig, ax = plt.subplots(figsize=(3.4, 1.45))
    y = range(len(labels))
    ax.barh(y, values, height=0.58, color=PLAIN, linewidth=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 52)
    ax.set_xlabel("Stage latency (ms)")
    tidy(ax)

    for i, v in enumerate(values):
        ax.text(v + 0.9, i, f"{v}", va="center", ha="left",
                fontsize=6.5, color=INK2)
    # The 82.2 ms total is stated in the caption rather than annotated here,
    # where it collided with the x-axis tick labels.
    save(fig, "fig_trace")


if __name__ == "__main__":
    print("Generating figures from the paper's own tables:")
    fig_differential()
    fig_latency()
    fig_trace()
    print("Done.")
