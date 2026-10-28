"""Recompute optimization, predictions and metrics without generation helpers."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import sqlite3
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, atol=1e-7, rtol=1e-9):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


def vector_logits(x, beta):
    return x @ np.asarray(beta)


def independent_weights(y, q=None):
    n_positive = int(np.sum(y == 1))
    n_negative = len(y) - n_positive
    return (
        np.ones(len(y)) / len(y)
        if q is None
        else np.array(
            [q / n_positive if value else (1 - q) / n_negative for value in y]
        )
    )


def loss(eta, y, w):
    # logaddexp is replaced with a scalar stable softplus identity.
    values = [
        max(float(z), 0) + math.log1p(math.exp(-abs(float(z)))) - int(t) * float(z)
        for z, t in zip(eta, y)
    ]
    return math.fsum(float(a) * b for a, b in zip(w, values))


def kkt_residual(beta, gradient, bounds):
    residual = []
    for b, g, (lower, upper) in zip(beta, gradient, bounds):
        residual.append(
            min(g, 0) if b <= lower + 1e-8 else max(g, 0) if b >= upper - 1e-8 else g
        )
    return max(abs(v) for v in residual)


def check_fit(fit, x, y, w, alpha, counters, bounds=None):
    values = []
    for attempt in fit["attempts"]:
        beta = np.array(attempt["coefficients"])
        if bounds is not None:
            assert all(
                lower <= value <= upper for value, (lower, upper) in zip(beta, bounds)
            )
        if "alpha" in fit:
            assert fit["alpha"] == alpha
        eta = vector_logits(x, beta)
        p = expit(eta)
        value = (
            loss(eta, y, w) + alpha * math.fsum(float(b) ** 2 for b in beta[:-1]) / 2
        )
        gradient = np.array(
            [
                math.fsum(float(v) * float(r) for v, r in zip(x[:, j], w * (p - y)))
                for j in range(x.shape[1])
            ]
        )
        gradient[:-1] += alpha * beta[:-1]
        hessian = x.T @ (x * (w * p * (1 - p))[:, None])
        hessian[np.arange(len(beta) - 1), np.arange(len(beta) - 1)] += alpha
        assert attempt["success"], attempt
        close(value, attempt["objective"])
        residual = (
            kkt_residual(beta, gradient, bounds)
            if bounds
            else float(np.max(abs(gradient)))
        )
        assert residual <= 1e-6, residual
        minimum_eigenvalue = float(np.linalg.eigvalsh((hessian + hessian.T) / 2)[0])
        assert minimum_eigenvalue >= -1e-10, minimum_eigenvalue
        direction = np.arange(1, len(beta) + 1, dtype=float)
        direction /= np.linalg.norm(direction)

        def objective_at(b):
            return (
                loss(vector_logits(x, b), y, w)
                + alpha * math.fsum(float(c) ** 2 for c in b[:-1]) / 2
            )

        delta = 1e-5
        finite = (
            objective_at(beta + delta * direction)
            - objective_at(beta - delta * direction)
        ) / (2 * delta)
        close(finite, float(gradient @ direction), atol=1e-5, rtol=1e-5)
        values.append(value)
        counters["optimizer_attempts"] += 1
        counters["maximum_kkt_residual"] = max(
            counters["maximum_kkt_residual"], residual
        )
        counters["minimum_hessian_eigenvalue"] = min(
            counters["minimum_hessian_eigenvalue"], minimum_eigenvalue
        )
    assert np.allclose(values, values[0], atol=1e-8, rtol=1e-8)
    expected_start = min(fit["attempts"], key=lambda a: (a["objective"], a["start_id"]))
    assert fit["chosen_start"] == expected_start["start_id"]
    if "coefficients" in fit:
        close(fit["coefficients"], expected_start["coefficients"])
    return np.array(expected_start["coefficients"])


def check_metric(row, eta, y, q):
    w = independent_weights(y, q)
    p = expit(eta)
    expected = {
        "log_loss": loss(eta, y, w),
        "brier_mse": math.fsum(
            float(a) * float(b - c) ** 2 for a, b, c in zip(w, p, y)
        ),
        "mean_probability": math.fsum(float(a * b) for a, b in zip(w, p)),
        "n_positive": int(sum(y)),
        "n_negative": int(sum(y == 0)),
    }
    expected["calibration_offset"] = expected["mean_probability"] - q
    # Compare logits against the equivalent probability threshold independently.
    alert = eta >= math.log(10)
    for key, truth, flag in [
        ("tp", 1, True),
        ("fp", 0, True),
        ("tn", 0, False),
        ("fn", 1, False),
    ]:
        expected[key] = math.fsum(
            float(ww)
            for yy, aa, ww in zip(y, alert, w)
            if yy == truth and bool(aa) == flag
        )
    expected["review_fraction"] = expected["tp"] + expected["fp"]
    expected["precision"] = (
        expected["tp"] / expected["review_fraction"]
        if expected["review_fraction"]
        else None
    )
    expected["recall"] = expected["tp"] / q
    expected["false_positive_rate"] = expected["fp"] / (1 - q)
    expected["decision_loss"] = 10 * expected["fp"] + expected["fn"]
    for key, value in expected.items():
        assert (
            row[key] is None
            if value is None
            else np.isclose(row[key], value, atol=1e-7, rtol=1e-9)
        ), (key, row[key], value)


def check_bootstrap(record, seed, frame, odds, refit, root):
    supplied = json.loads(
        (root / f"results/bootstrap/{record['id']}_{seed}.json").read_bytes()
    )
    groups = sorted(frame.group_id.unique())
    assert groups == supplied["ordered_groups"]
    y = frame.target.to_numpy(int)
    delta = np.array(
        [
            max(float(a), 0)
            + math.log1p(math.exp(-abs(float(a))))
            - int(t) * float(a)
            - max(float(b), 0)
            - math.log1p(math.exp(-abs(float(b))))
            + int(t) * float(b)
            for a, b, t in zip(odds, refit, y)
        ]
    )
    # Every current group has one row, but preserve full groups if a future
    # authorized panel contains multiple differently labelled members.
    members = {g: np.where(frame.group_id.to_numpy() == g)[0] for g in groups}
    negative_sums = np.array(
        [math.fsum(float(delta[i]) for i in members[g] if y[i] == 0) for g in groups]
    )
    positive_sums = np.array(
        [math.fsum(float(delta[i]) for i in members[g] if y[i] == 1) for g in groups]
    )
    negative_counts = np.array([sum(y[members[g]] == 0) for g in groups])
    positive_counts = np.array([sum(y[members[g]] == 1) for g in groups])
    rng = np.random.default_rng(20261028 + record["id"] + seed)
    digest = hashlib.sha256()
    differences = []
    for entry in supplied["draws"]:
        sample = rng.integers(0, len(groups), size=len(groups), dtype=np.int64)
        digest.update(sample.astype("<i8").tobytes())
        n0 = int(sum(negative_counts[sample]))
        n1 = int(sum(positive_counts[sample]))
        assert (n0, n1) == (entry["n_negative"], entry["n_positive"])
        if not n0 or not n1:
            assert not entry["valid"] and entry["difference"] is None
        else:
            actual = (
                0.99 * math.fsum(float(v) for v in negative_sums[sample]) / n0
                + 0.01 * math.fsum(float(v) for v in positive_sums[sample]) / n1
            )
            close(actual, entry["difference"])
            differences.append(actual)
    assert digest.hexdigest() == supplied["draw_index_sha256"]
    if len(differences) == 1000:
        close(np.quantile(differences, [0.025, 0.975]), supplied["interval_95"])
    else:
        assert supplied["interval_95"] is None
    return len(supplied["draws"])


def verify(root, stage):
    counters = {
        "optimizer_attempts": 0,
        "maximum_kkt_residual": 0.0,
        "minimum_hessian_eigenvalue": float("inf"),
        "validation_predictions": 0,
        "test_predictions": 0,
        "test_metric_rows": 0,
        "bootstrap_draws": 0,
    }
    val_predictions = pd.read_csv(
        root / "results/validation_predictions.csv.gz", float_precision="round_trip"
    )
    split_file = pd.read_csv(root / "results/splits.csv.gz")
    metrics = (
        json.loads((root / "results/test_metrics.json").read_bytes())
        if stage == "test"
        else []
    )
    test_predictions = (
        pd.read_csv(
            root / "results/test_predictions.csv.gz", float_precision="round_trip"
        )
        if stage == "test"
        else None
    )
    for record in json.loads((root / "data/manifest.json").read_bytes()):
        frame = pd.read_csv(
            root / record["panel_file"],
            float_precision="round_trip",
            dtype={"source_label": str},
        )
        x = frame[record["features"]].to_numpy(float)
        y = frame.target.to_numpy(int)
        for seed in [101, 202, 303, 404, 505]:
            bucket = np.array(
                [
                    int(
                        hashlib.sha256((g + ":" + str(seed)).encode()).hexdigest()[:8],
                        16,
                    )
                    % 20
                    for g in frame.group_id
                ]
            )
            part = np.array(
                [
                    (
                        "fit"
                        if v < 10
                        else (
                            "validation"
                            if v < 13
                            else "calibration" if v < 16 else "test"
                        )
                    )
                    for v in bucket
                ]
            )
            split = split_file[
                (split_file.dataset == record["id"]) & (split_file.seed == seed)
            ]
            assert (
                split.partition.tolist() == part.tolist()
                and split.source_row.tolist() == frame.source_row.tolist()
            )
            connection = sqlite3.connect(":memory:")
            split.to_sql("split", connection, index=False)
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM (SELECT group_id FROM split GROUP BY group_id HAVING COUNT(DISTINCT partition)>1)"
                ).fetchone()[0]
                == 0
            )
            counts = connection.execute(
                "SELECT partition,target,COUNT(*) FROM split GROUP BY partition,target"
            ).fetchall()
            assert len(counts) == 8 and min(v[2] for v in counts) >= 20
            model = json.loads(
                (root / f"results/models/{record['id']}_{seed}.json").read_bytes()
            )
            assert [c["alpha"] for c in model["candidates"]] == [
                0.0001,
                0.001,
                0.01,
                0.1,
                1.0,
                10.0,
            ]
            fit = part == "fit"
            val = part == "validation"
            cal = part == "calibration"
            held = part == "test"
            basefit = fit | val
            allfit = basefit | cal
            mean = x[fit].mean(axis=0)
            scale = np.sqrt(((x[fit] - mean) ** 2).mean(axis=0))
            scale[scale == 0] = 1
            close(model["fit_mean"], mean)
            close(model["fit_scale"], scale)

            def matrix(mask, m, s):
                return np.column_stack(((x[mask] - m) / s, np.ones(sum(mask))))

            for candidate in model["candidates"]:
                beta = check_fit(
                    candidate,
                    matrix(fit, mean, scale),
                    y[fit],
                    independent_weights(y[fit]),
                    candidate["alpha"],
                    counters,
                )
                eta = vector_logits(matrix(val, mean, scale), beta)
                close(
                    loss(eta, y[val], independent_weights(y[val])),
                    candidate["validation_log_loss"],
                )
                exported = val_predictions[
                    (val_predictions.dataset == record["id"])
                    & (val_predictions.seed == seed)
                    & (val_predictions.alpha == candidate["alpha"])
                ]
                assert (
                    exported.source_row.tolist() == frame.loc[val].source_row.tolist()
                )
                close(exported.logit.to_numpy(), eta)
                counters["validation_predictions"] += len(eta)
            selected = min(
                model["candidates"],
                key=lambda c: (c["validation_log_loss"], c["alpha"]),
            )
            assert model["selected_alpha"] == selected["alpha"]
            bm = x[basefit].mean(axis=0)
            bs = np.sqrt(((x[basefit] - bm) ** 2).mean(axis=0))
            bs[bs == 0] = 1
            close(model["base_mean"], bm)
            close(model["base_scale"], bs)
            beta = check_fit(
                model["base"],
                matrix(basefit, bm, bs),
                y[basefit],
                independent_weights(y[basefit]),
                model["selected_alpha"],
                counters,
            )
            cal_eta = vector_logits(matrix(cal, bm, bs), beta)
            if stage == "test":
                held_frame = frame.loc[held].reset_index(drop=True)
                held_eta = vector_logits(matrix(held, bm, bs), beta)
            for adaptation in model["adaptations"]:
                q = adaptation["prior"]
                w = independent_weights(y[cal], q)
                close(sum(w), 1)
                close(sum(w * y[cal]), q)
                odds = math.log(q / (1 - q)) - math.log(
                    float(np.mean(y[basefit])) / (1 - float(np.mean(y[basefit])))
                )
                close(odds, adaptation["odds_offset"])
                b = adaptation["intercept"]["offset"]
                derivative = math.fsum(
                    float(a * (expit(z + b) - yy))
                    for a, z, yy in zip(w, cal_eta, y[cal])
                )
                assert (
                    -20 <= b <= 20
                    and kkt_residual([b], [derivative], [(-20, 20)]) <= 1e-6
                )
                close(
                    loss(cal_eta + b, y[cal], w), adaptation["intercept"]["objective"]
                )
                platt = adaptation["slope_intercept"]
                coefficients = check_fit(
                    platt,
                    np.column_stack((cal_eta, np.ones(len(cal_eta)))),
                    y[cal],
                    w,
                    0,
                    counters,
                    bounds=[(0, 5), (-20, 20)],
                )
                close(coefficients, [platt["slope"], platt["offset"]])
                refit = check_fit(
                    adaptation["coefficient_refit"],
                    matrix(allfit, bm, bs),
                    y[allfit],
                    independent_weights(y[allfit], q),
                    model["selected_alpha"],
                    counters,
                )
                if stage == "test":
                    logits = {
                        "unchanged": held_eta,
                        "prior_odds": held_eta + odds,
                        "intercept": held_eta + b,
                        "slope_intercept": held_eta * coefficients[0] + coefficients[1],
                        "coefficient_refit": vector_logits(matrix(held, bm, bs), refit),
                        "known_prior": np.full(sum(held), math.log(q / (1 - q))),
                    }
                    for method, eta in logits.items():
                        row = [
                            r
                            for r in metrics
                            if (r["dataset"], r["seed"], r["prior"], r["method"])
                            == (record["id"], seed, q, method)
                        ]
                        assert len(row) == 1
                        check_metric(row[0], eta, y[held], q)
                        counters["test_metric_rows"] += 1
                        exported = test_predictions[
                            (test_predictions.dataset == record["id"])
                            & (test_predictions.seed == seed)
                            & (test_predictions.prior == q)
                            & (test_predictions.method == method)
                        ]
                        assert (
                            exported.source_row.tolist()
                            == held_frame.source_row.tolist()
                        )
                        close(exported.logit.to_numpy(), eta)
                        close(exported.probability.to_numpy(), expit(eta))
                        close(
                            exported.weight.to_numpy(), independent_weights(y[held], q)
                        )
                        counters["test_predictions"] += len(eta)
                    if q == 0.01:
                        counters["bootstrap_draws"] += check_bootstrap(
                            record,
                            seed,
                            held_frame,
                            logits["prior_odds"],
                            logits["coefficient_refit"],
                            root,
                        )
            print(
                f"Independent {stage} check PASS: {record['id']} seed{seed}", flush=True
            )
    if stage == "test":
        assert counters["test_metric_rows"] == len(metrics) == 240
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "results/models").glob("*.json"))
    }
    receipt = {
        "status": "PASS",
        "stage": stage,
        "model_hashes": hashes,
        "independent_methods": "Scalar softplus/fsum, matrix products, directional derivatives, Hessian eigenvalues, SQL group checks and direct resampling",
        **counters,
    }
    (root / f"results/INDEPENDENT_{stage.upper()}_CHECK.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "test"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.root, args.stage)
