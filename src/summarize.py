"""Describe all prespecified comparisons, including ineffective decisions."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
metrics = pd.DataFrame(json.loads((ROOT / "results/test_metrics.json").read_bytes()))
selection = json.loads((ROOT / "results/selection.json").read_bytes())
medians = []
for (dataset, prior, method), rows in metrics.groupby(
    ["dataset", "prior", "method"], sort=True
):
    medians.append(
        {
            "dataset": int(dataset),
            "prior": float(prior),
            "method": method,
            "split_count": len(rows),
            **{
                key: float(rows[key].median())
                for key in [
                    "log_loss",
                    "brier_mse",
                    "decision_loss",
                    "recall",
                    "false_positive_rate",
                    "review_fraction",
                ]
            },
            "no_alert_splits": int((rows.review_fraction == 0).sum()),
            "decision_loss_above_no_alert_splits": int(
                (rows.decision_loss > prior + 1e-12).sum()
            ),
        }
    )
pairs = []
for dataset in [44125, 44128]:
    held = metrics[(metrics.dataset == dataset) & (metrics.prior == 0.01)]
    odds = held[held.method == "prior_odds"].set_index("seed")
    refit = held[held.method == "coefficient_refit"].set_index("seed")
    difference = odds.log_loss - refit.log_loss
    pairs.append(
        {
            "dataset": dataset,
            "prior": 0.01,
            "odds_log_loss_wins": int((difference < -1e-12).sum()),
            "ties": int((abs(difference) <= 1e-12).sum()),
            "refit_log_loss_wins": int((difference > 1e-12).sum()),
            "median_paired_log_loss_odds_minus_refit": float(np.median(difference)),
            "minimum_paired_difference": float(min(difference)),
            "maximum_paired_difference": float(max(difference)),
        }
    )
boundary = []
for path in sorted((ROOT / "results/models").glob("*.json")):
    model = json.loads(path.read_bytes())
    for item in model["adaptations"]:
        platt = item["slope_intercept"]
        boundary.append(
            {
                "dataset": model["dataset"],
                "seed": model["seed"],
                "prior": item["prior"],
                "slope": platt["slope"],
                "offset": platt["offset"],
                "slope_boundary": platt["slope"] <= 1e-8 or platt["slope"] >= 5 - 1e-8,
                "offset_boundary": abs(platt["offset"]) >= 20 - 1e-8,
                "intercept_boundary": item["intercept"]["boundary"],
            }
        )
summary = {
    "scope": "Controlled known-prior scenarios; five overlapping group splits per benchmark, not independent deployment replications",
    "medians": medians,
    "rare_paired_comparisons": pairs,
    "base_alpha_boundary_winners": sum(row["boundary_alpha"] for row in selection),
    "base_model_count": len(selection),
    "adaptation_boundaries": boundary,
    "decision_units": "Hypothetical FP10/FN1 loss per scenario observation; no-alert reference=q; not currency",
    "no_claim": "No universal replacement, new calibration algorithm, measured production drift, business savings or actual detector operating cost",
}
(ROOT / "results/summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "rare_pairs": pairs,
            "alpha_boundary_winners": summary["base_alpha_boundary_winners"],
            "platt_active_boundaries": sum(
                r["slope_boundary"] or r["offset_boundary"] for r in boundary
            ),
        }
    ),
    flush=True,
)
