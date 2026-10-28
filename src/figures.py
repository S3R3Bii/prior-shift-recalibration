"""Decision-useful graphics from frozen, independently checked outputs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
METRICS = pd.DataFrame(json.loads((ROOT / "results/test_metrics.json").read_bytes()))
METHODS = [
    "unchanged",
    "prior_odds",
    "intercept",
    "slope_intercept",
    "coefficient_refit",
    "known_prior",
]
LABELS = [
    "Unchanged",
    "Prior odds",
    "Intercept",
    "Slope + intercept",
    "Coefficient refit",
    "Known-prior constant",
]
COLORS = ["#A54B32", "#197F85", "#5797A3", "#7D739F", "#234E77", "#7D838A"]
NAMES = {44125: "MAGIC · Monte Carlo simulation", 44128: "MiniBooNE · event benchmark"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "#FAFAF7",
        "axes.facecolor": "#FAFAF7",
        "axes.labelcolor": "#28323D",
        "text.color": "#28323D",
    }
)


def finish(fig, name, note):
    fig.text(0.07, 0.035, note, fontsize=9, va="bottom", linespacing=1.4)
    fig.savefig(OUT / (name + ".png"), dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(OUT / (name + ".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.5), sharey=True)
fig.subplots_adjust(left=0.18, right=0.97, top=0.78, bottom=0.23, wspace=0.16)
fig.suptitle(
    "Correcting prevalence helps scores; replacing refits is less clear",
    x=0.07,
    y=0.96,
    ha="left",
    fontsize=17,
    fontweight="bold",
)
fig.text(
    0.07,
    0.89,
    "CONTROLLED PRIOR-SHIFT SCENARIOS  |  N1 corrected retrospective  |  supplied prevalence q = 1%",
    fontsize=10,
    color="#56616D",
)
for ax, dataset in zip(axes, [44125, 44128]):
    subset = METRICS[(METRICS.dataset == dataset) & (METRICS.prior == 0.01)]
    for i, method in enumerate(METHODS):
        values = subset[subset.method == method].sort_values("seed").log_loss.to_numpy()
        ax.scatter(
            values,
            i + np.linspace(-0.13, 0.13, len(values)),
            c=COLORS[i],
            s=28,
            alpha=0.7,
            zorder=3,
        )
        ax.plot(
            [np.median(values), np.median(values)],
            [i - 0.28, i + 0.28],
            color=COLORS[i],
            linewidth=3,
        )
    ax.set_title(NAMES[dataset], fontsize=11, pad=13)
    ax.set_xscale("log")
    ax.set_xlim(0.025, 0.65)
    ax.set_xticks([0.03, 0.05, 0.1, 0.2, 0.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_yticks(range(6), LABELS)
    ax.set_ylim(5.5, -0.5)
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlabel("Weighted test log loss (nats; log axis)\nLower is better")
finish(
    fig,
    "rare_probability_scores",
    "Dots: five overlapping group splits; bars: within-method median. Original OpenML panels are bounded derivatives.\nThe known target prior is supplied to every adaptation. These are not measured production drift or independent replications.",
)

comparisons = pd.DataFrame(
    json.loads((ROOT / "results/rare_comparisons.json").read_bytes())
)
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.5), sharex=True, sharey=True)
fig.subplots_adjust(left=0.10, right=0.97, top=0.78, bottom=0.23, wspace=0.19)
fig.suptitle(
    "The small refit-versus-odds gap changes across splits",
    x=0.07,
    y=0.96,
    ha="left",
    fontsize=17,
    fontweight="bold",
)
fig.text(
    0.07,
    0.89,
    "CONTROLLED PRIOR-SHIFT SCENARIOS  |  N1 corrected retrospective  |  q = 1% paired log-loss difference",
    fontsize=10,
    color="#56616D",
)
for ax, dataset in zip(axes, [44125, 44128]):
    rows = comparisons[comparisons.dataset == dataset].sort_values("seed")
    for i, row in enumerate(rows.itertuples()):
        lower, upper = row.interval_95
        ax.hlines(i, lower, upper, color="#197F85", linewidth=2)
        ax.scatter(row.log_loss_odds_minus_refit, i, color="#234E77", s=45, zorder=4)
    ax.axvline(0, color="#6F757B", linewidth=1, linestyle="--")
    ax.set_title(NAMES[dataset], fontsize=11, pad=13)
    ax.set_yticks(range(5), [f"Seed {s}" for s in rows.seed])
    ax.set_ylim(4.5, -0.5)
    ax.set_xlim(-0.006, 0.005)
    ax.set_xticks([-0.004, -0.002, 0, 0.002, 0.004])
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlabel(
        "Prior-odds loss − coefficient-refit loss (nats)\nNegative favors odds; positive favors refit"
    )
finish(
    fig,
    "paired_refit_gap",
    "Lines: conditional 95% percentile intervals from 1,000 whole-feature-group draws per split. No pooled significance test.\nSplit seeds overlap. Intervals assume these benchmark groups and the supplied target prevalence; unknown run dependence remains.",
)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), sharey=True)
fig.subplots_adjust(left=0.16, right=0.86, top=0.77, bottom=0.24, wspace=0.18)
fig.suptitle(
    "Better probabilities do not automatically create useful alerts",
    x=0.07,
    y=0.96,
    ha="left",
    fontsize=17,
    fontweight="bold",
)
fig.text(
    0.07,
    0.89,
    "CONTROLLED PRIOR-SHIFT SCENARIOS  |  N1 corrected retrospective  |  FP cost 10, FN cost 1; alert p ≥ 10/11",
    fontsize=10,
    color="#56616D",
)
norm = LogNorm(vmin=0.8, vmax=25)
for ax, dataset in zip(axes, [44125, 44128]):
    values = np.array(
        [
            [
                METRICS[
                    (METRICS.dataset == dataset)
                    & (METRICS.method == method)
                    & (METRICS.prior == q)
                ].decision_loss.median()
                / q
                for q in [0.01, 0.05, 0.2, 0.5]
            ]
            for method in METHODS
        ]
    )
    image = ax.imshow(values, cmap="cividis_r", norm=norm, aspect="auto")
    for i in range(6):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{values[i,j]:.2f}×",
                ha="center",
                va="center",
                color="white" if norm(values[i, j]) > 0.55 else "#182D39",
                fontsize=10,
            )
    ax.set_title(NAMES[dataset], fontsize=11, pad=13)
    ax.set_xticks(range(4), ["1%", "5%", "20%", "50%"])
    ax.set_yticks(range(6), LABELS)
    ax.set_xlabel("Supplied positive prevalence")
    ax.tick_params(length=0)
bar_ax = fig.add_axes([0.89, 0.25, 0.018, 0.51])
bar = fig.colorbar(image, cax=bar_ax, ticks=[1, 2, 5, 10, 20])
bar.ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×"])
bar.set_label("Median loss / no-alert loss (log color scale)", labelpad=10)
finish(
    fig,
    "decision_loss_tradeoff",
    "1× equals flagging nothing; above 1× is worse under this hypothetical cost. Cells show medians across five overlapping splits.\nMAGIC adaptations flag no events at q ≤ 20%. Probability improvements alone do not establish detector utility or financial savings.",
)
print("Saved three controlled-scenario figures in PNG/PDF", flush=True)
