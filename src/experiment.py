"""Separate training/freeze from test scoring for known-prior scenarios."""

from pathlib import Path
import argparse
import gzip
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from model import (
    ALPHAS,
    SEEDS,
    PRIORS,
    METHODS,
    partitions,
    weights,
    standardize,
    design,
    linear,
    solve_ridge,
    solve_intercept,
    solve_platt,
    metrics,
    OptimizationFailure,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0))


def panels(root):
    for record in json.loads((root / "data/manifest.json").read_bytes()):
        frame = pd.read_csv(
            root / record["panel_file"],
            float_precision="round_trip",
            dtype={"source_label": str},
        )
        yield record, frame, frame[record["features"]].to_numpy(
            float
        ), frame.target.to_numpy(int)


def train(root):
    rows = []
    validation_predictions = []
    splits = []
    for record, frame, x, y in panels(root):
        for seed in SEEDS:
            part = partitions(frame.group_id, seed)
            split = frame[["source_row", "group_id", "target"]].copy()
            split["dataset"] = record["id"]
            split["seed"] = seed
            split["partition"] = part
            splits.append(split)
            for name in ["fit", "validation", "calibration", "test"]:
                assert min(np.bincount(y[part == name], minlength=2)) >= 20
            fit = part == "fit"
            val = part == "validation"
            base_fit = fit | val
            cal = part == "calibration"
            all_fit = base_fit | cal
            mean, scale = standardize(x[fit])
            fit_x = design(x[fit], mean, scale)
            val_x = design(x[val], mean, scale)
            candidates = []
            for alpha in ALPHAS:
                candidate = solve_ridge(fit_x, y[fit], weights(y[fit]), alpha)
                eta = linear(val_x, np.array(candidate["coefficients"]))
                candidate["validation_log_loss"] = float(
                    np.mean(np.logaddexp(0, eta) - y[val] * eta)
                )
                candidates.append(candidate)
                pred = frame.loc[val, ["source_row", "group_id", "target"]].copy()
                pred["dataset"] = record["id"]
                pred["seed"] = seed
                pred["alpha"] = alpha
                pred["logit"] = eta
                validation_predictions.append(pred)
            selected = min(
                candidates, key=lambda c: (c["validation_log_loss"], c["alpha"])
            )
            base_mean, base_scale = standardize(x[base_fit])
            base_x = design(x[base_fit], base_mean, base_scale)
            base = solve_ridge(
                base_x, y[base_fit], weights(y[base_fit]), selected["alpha"]
            )
            base_prevalence = float(np.mean(y[base_fit]))
            cal_eta = linear(
                design(x[cal], base_mean, base_scale), np.array(base["coefficients"])
            )
            adaptations = []
            for q in PRIORS:
                odds = float(
                    np.log(q / (1 - q))
                    - np.log(base_prevalence / (1 - base_prevalence))
                )
                intercept = solve_intercept(cal_eta, y[cal], weights(y[cal], q))
                platt = solve_platt(cal_eta, y[cal], weights(y[cal], q), odds)
                refit = solve_ridge(
                    design(x[all_fit], base_mean, base_scale),
                    y[all_fit],
                    weights(y[all_fit], q),
                    selected["alpha"],
                )
                adaptations.append(
                    {
                        "prior": q,
                        "odds_offset": odds,
                        "intercept": intercept,
                        "slope_intercept": platt,
                        "coefficient_refit": refit,
                    }
                )
            model = {
                "dataset": record["id"],
                "seed": seed,
                "fit_mean": mean.tolist(),
                "fit_scale": scale.tolist(),
                "candidates": candidates,
                "selected_alpha": selected["alpha"],
                "boundary_alpha": selected["alpha"] in (ALPHAS[0], ALPHAS[-1]),
                "base_mean": base_mean.tolist(),
                "base_scale": base_scale.tolist(),
                "base_prevalence": base_prevalence,
                "base": base,
                "adaptations": adaptations,
            }
            write_json(root / f"results/models/{record['id']}_{seed}.json", model)
            rows.append(
                {
                    "dataset": record["id"],
                    "seed": seed,
                    "selected_alpha": selected["alpha"],
                    "validation_log_loss": selected["validation_log_loss"],
                    "boundary_alpha": model["boundary_alpha"],
                }
            )
            print(
                json.dumps(
                    {
                        "trained_dataset": record["id"],
                        "seed": seed,
                        "selected_alpha": selected["alpha"],
                    }
                ),
                flush=True,
            )
    write_csv(root / "results/splits.csv.gz", pd.concat(splits, ignore_index=True))
    write_csv(
        root / "results/validation_predictions.csv.gz",
        pd.concat(validation_predictions, ignore_index=True),
    )
    write_json(root / "results/selection.json", rows)
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "results/models").glob("*.json"))
    }
    write_json(
        root / "results/freeze.json",
        {
            "state": "CORRECTED_RETROSPECTIVE_FROZEN_BEFORE_RESCORING",
            "test_reuse": True,
            "correction": "N1 numerical convergence repair after original test and failed coefficient replay; selection grid, splits, costs and tolerances unchanged",
            "model_hashes": hashes,
            "selection_sha256": hashlib.sha256(
                (root / "results/selection.json").read_bytes()
            ).hexdigest(),
            "test_target_use": "None in any base/adaptation objective, preprocessing or selection",
        },
    )


def bootstrap(frame, odds_eta, refit_eta, seed):
    groups = sorted(frame.group_id.unique())
    mapping = {g: i for i, g in enumerate(groups)}
    index = np.array([mapping[g] for g in frame.group_id])
    y = frame.target.to_numpy(int)
    difference = (
        np.logaddexp(0, odds_eta)
        - y * odds_eta
        - (np.logaddexp(0, refit_eta) - y * refit_eta)
    )
    sums = [
        np.bincount(index[y == c], weights=difference[y == c], minlength=len(groups))
        for c in [0, 1]
    ]
    counts = [np.bincount(index[y == c], minlength=len(groups)) for c in [0, 1]]
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    draws = []
    for draw in range(1000):
        sampled = rng.integers(0, len(groups), len(groups), dtype=np.int64)
        digest.update(sampled.astype("<i8").tobytes())
        multiplicity = np.bincount(sampled, minlength=len(groups))
        n0 = int(np.sum(multiplicity * counts[0]))
        n1 = int(np.sum(multiplicity * counts[1]))
        value = (
            float(
                0.99 * np.sum(multiplicity * sums[0]) / n0
                + 0.01 * np.sum(multiplicity * sums[1]) / n1
            )
            if n0 and n1
            else None
        )
        draws.append(
            {
                "draw": draw,
                "n_negative": n0,
                "n_positive": n1,
                "difference": value,
                "valid": value is not None,
            }
        )
    valid = all(d["valid"] for d in draws)
    interval = (
        np.quantile([d["difference"] for d in draws], [0.025, 0.975]).tolist()
        if valid
        else None
    )
    return {
        "seed": seed,
        "rng": "numpy.default_rng.PCG64",
        "numpy_version": np.__version__,
        "ordered_groups": groups,
        "draw_index_sha256": digest.hexdigest(),
        "draws": draws,
        "interval_95": interval,
        "interval_status": (
            "CONDITIONAL_BENCHMARK_INTERVAL"
            if valid
            else "UNVERIFIED_INVALID_CLASS_DRAW"
        ),
    }


def test(root):
    freeze = json.loads((root / "results/freeze.json").read_bytes())
    assert freeze["state"] == "CORRECTED_RETROSPECTIVE_FROZEN_BEFORE_RESCORING"
    certificate = json.loads(
        (root / "results/INDEPENDENT_TRAIN_CHECK.json").read_bytes()
    )
    assert (
        certificate["status"] == "PASS"
        and certificate["model_hashes"] == freeze["model_hashes"]
    )
    assert (
        hashlib.sha256((root / "results/selection.json").read_bytes()).hexdigest()
        == freeze["selection_sha256"]
    )
    for path, digest in freeze["model_hashes"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    rows = []
    predictions = []
    comparisons = []
    for record, frame, x, y in panels(root):
        for seed in SEEDS:
            model = json.loads(
                (root / f"results/models/{record['id']}_{seed}.json").read_bytes()
            )
            held = partitions(frame.group_id, seed) == "test"
            held_frame = frame.loc[held].reset_index(drop=True)
            xx = design(
                x[held], np.array(model["base_mean"]), np.array(model["base_scale"])
            )
            base_eta = linear(xx, np.array(model["base"]["coefficients"]))
            for adaptation in model["adaptations"]:
                q = adaptation["prior"]
                w = weights(y[held], q)
                logits = {
                    "unchanged": base_eta,
                    "prior_odds": base_eta + adaptation["odds_offset"],
                    "intercept": base_eta + adaptation["intercept"]["offset"],
                    "slope_intercept": base_eta * adaptation["slope_intercept"]["slope"]
                    + adaptation["slope_intercept"]["offset"],
                    "coefficient_refit": linear(
                        xx, np.array(adaptation["coefficient_refit"]["coefficients"])
                    ),
                    "known_prior": np.full(np.sum(held), np.log(q / (1 - q))),
                }
                for method in METHODS:
                    eta = logits[method]
                    rows.append(
                        {
                            "dataset": record["id"],
                            "seed": seed,
                            "prior": q,
                            "method": method,
                            **metrics(y[held], eta, w, q),
                        }
                    )
                    pred = held_frame[["source_row", "group_id", "target"]].copy()
                    pred["dataset"] = record["id"]
                    pred["seed"] = seed
                    pred["prior"] = q
                    pred["method"] = method
                    pred["logit"] = eta
                    pred["probability"] = expit(eta)
                    pred["weight"] = w
                    predictions.append(pred)
                if q == 0.01:
                    result = bootstrap(
                        held_frame,
                        logits["prior_odds"],
                        logits["coefficient_refit"],
                        20261028 + record["id"] + seed,
                    )
                    write_json(
                        root / f"results/bootstrap/{record['id']}_{seed}.json", result
                    )
                    a = metrics(y[held], logits["prior_odds"], w, q)["log_loss"]
                    b = metrics(y[held], logits["coefficient_refit"], w, q)["log_loss"]
                    comparisons.append(
                        {
                            "dataset": record["id"],
                            "seed": seed,
                            "log_loss_odds_minus_refit": a - b,
                            "interval_95": result["interval_95"],
                            "valid_draws": sum(d["valid"] for d in result["draws"]),
                        }
                    )
    write_csv(
        root / "results/test_predictions.csv.gz",
        pd.concat(predictions, ignore_index=True),
    )
    write_json(root / "results/test_metrics.json", rows)
    write_json(root / "results/rare_comparisons.json", comparisons)
    write_json(
        root / "results/runtime.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "thread_limit": 1,
            "source_scope": "Controlled known-prior scenarios, not observed deployment drift",
        },
    )
    print(
        json.dumps(
            {
                "metric_rows": len(rows),
                "prediction_rows": sum(len(p) for p in predictions),
                "bootstrap_draws": 1000 * len(comparisons),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "test"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        try:
            (train if args.stage == "train" else test)(args.root)
        except OptimizationFailure as error:
            from datetime import datetime, timezone

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            write_json(
                args.root / f"results/failed_attempts/{stamp}.json", error.evidence
            )
            raise
