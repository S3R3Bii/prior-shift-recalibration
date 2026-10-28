"""Independent scalar aggregation of every displayed median and paired result."""

from pathlib import Path
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results/test_metrics.json").read_bytes())
summary = json.loads((ROOT / "results/summary.json").read_bytes())
comparisons = json.loads((ROOT / "results/rare_comparisons.json").read_bytes())
selection = json.loads((ROOT / "results/selection.json").read_bytes())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7), (a, b)


for item in summary["medians"]:
    group = [
        r
        for r in rows
        if (r["dataset"], r["prior"], r["method"])
        == (item["dataset"], item["prior"], item["method"])
    ]
    assert len(group) == item["split_count"] == 5
    for key in [
        "log_loss",
        "brier_mse",
        "decision_loss",
        "recall",
        "false_positive_rate",
        "review_fraction",
    ]:
        close(statistics.median(r[key] for r in group), item[key])
    assert sum(r["review_fraction"] == 0 for r in group) == item["no_alert_splits"]
    assert (
        sum(r["decision_loss"] > item["prior"] + 1e-12 for r in group)
        == item["decision_loss_above_no_alert_splits"]
    )
for item in comparisons:

    def metric(method):
        return next(
            r
            for r in rows
            if (r["dataset"], r["seed"], r["prior"], r["method"])
            == (item["dataset"], item["seed"], 0.01, method)
        )["log_loss"]

    close(
        metric("prior_odds") - metric("coefficient_refit"),
        item["log_loss_odds_minus_refit"],
    )
    draw = json.loads(
        (ROOT / f"results/bootstrap/{item['dataset']}_{item['seed']}.json").read_bytes()
    )
    assert (
        draw["interval_95"] == item["interval_95"]
        and sum(d["valid"] for d in draw["draws"]) == item["valid_draws"]
    )
for item in summary["rare_paired_comparisons"]:
    values = [
        r["log_loss_odds_minus_refit"]
        for r in comparisons
        if r["dataset"] == item["dataset"]
    ]
    close(statistics.median(values), item["median_paired_log_loss_odds_minus_refit"])
    close(min(values), item["minimum_paired_difference"])
    close(max(values), item["maximum_paired_difference"])
    assert sum(v < -1e-12 for v in values) == item["odds_log_loss_wins"]
    assert sum(v > 1e-12 for v in values) == item["refit_log_loss_wins"]
assert (
    sum(r["boundary_alpha"] for r in selection)
    == summary["base_alpha_boundary_winners"]
)
for item in summary["adaptation_boundaries"]:
    model = json.loads(
        (ROOT / f"results/models/{item['dataset']}_{item['seed']}.json").read_bytes()
    )
    adaptation = next(a for a in model["adaptations"] if a["prior"] == item["prior"])
    assert (
        adaptation["slope_intercept"]["slope"] == item["slope"]
        and adaptation["slope_intercept"]["offset"] == item["offset"]
    )
    assert item["slope_boundary"] == (
        item["slope"] <= 1e-8 or item["slope"] >= 5 - 1e-8
    )
    assert item["offset_boundary"] == (abs(item["offset"]) >= 20 - 1e-8)
    assert item["intercept_boundary"] == adaptation["intercept"]["boundary"]
receipt = {
    "status": "PASS",
    "summary_medians": len(summary["medians"]),
    "paired_results": len(comparisons),
    "adaptation_boundary_rows": len(summary["adaptation_boundaries"]),
    "no_pooled_seed_significance_claim": True,
}
(ROOT / "results/INDEPENDENT_SUMMARY_CHECK.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt), flush=True)
