"""Small adversarial contracts for the protected fitting and decision logic."""

from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model
import experiment
from verify import kkt_residual


class Contracts(unittest.TestCase):
    def test_whole_group_partition_and_no_target_input(self):
        groups = ["same", "same", "other"]
        parts = model.partitions(groups, 101)
        self.assertEqual(parts[0], parts[1])
        self.assertEqual(parts.tolist(), model.partitions(groups, 101).tolist())

    def test_prior_weights_and_odds_identity(self):
        y = np.array([0, 0, 0, 1, 1])
        for q in model.PRIORS:
            w = model.weights(y, q)
            self.assertAlmostEqual(sum(w), 1)
            self.assertAlmostEqual(sum(w * y), q)
        pi = float(np.mean(y))
        eta = np.array([-2.0, 0, 3.0])
        np.testing.assert_array_equal(
            eta + math.log(pi / (1 - pi)) - math.log(pi / (1 - pi)), eta
        )

    def test_extreme_logit_metrics_are_finite(self):
        row = model.metrics(
            np.array([1, 0]), np.array([-1000.0, 1000.0]), np.array([0.5, 0.5]), 0.5
        )
        self.assertEqual(row["log_loss"], 1000)
        self.assertEqual(row["brier_mse"], 1)
        self.assertEqual(row["decision_loss"], 5.5)

    def test_gradient_matches_central_difference_and_box_signs(self):
        x = np.array([[1, 1], [-2, 1], [3, 1]], dtype=float)
        y = np.array([1, 0, 0])
        w = model.weights(y, 0.2)
        beta = np.array([0.2, -0.4])
        _, grad = model.objective(beta, x, y, w, 0.1)
        for j in range(2):
            offset = np.eye(2)[j] * 1e-5
            numerical = (
                model.objective(beta + offset, x, y, w, 0.1)[0]
                - model.objective(beta - offset, x, y, w, 0.1)[0]
            ) / 2e-5
            self.assertAlmostEqual(numerical, grad[j], places=8)
        self.assertEqual(kkt_residual([0, 5], [3, -2], [(0, 5), (0, 5)]), 0)
        self.assertEqual(kkt_residual([0], [-3], [(0, 5)]), 3)

    def test_failed_optimizer_attempt_is_rejected(self):
        class Failed:
            x = np.array([0.0, 0.0])
            success = False
            status = 2
            message = "forced failure"
            nit = 0
            nfev = 0

        with patch.object(model, "minimize", return_value=Failed()):
            with self.assertRaises(model.OptimizationFailure) as caught:
                model.solve_ridge(
                    np.array([[1.0, 1.0], [-1.0, 1.0]]),
                    np.array([0, 1]),
                    np.array([0.5, 0.5]),
                    0.1,
                )
        self.assertEqual(len(caught.exception.evidence["attempts"]), 2)

    def test_newton_polish_preserves_objective_and_certifies_stationarity(self):
        x = np.column_stack((np.linspace(-2, 2, 100), np.ones(100)))
        y = (np.arange(100) % 3 == 0).astype(int); w = model.weights(y)
        beta, trace = model.newton_polish(np.array([.1, -.2]), x, y, w, .001)
        self.assertLessEqual(max(abs(model.objective(beta, x, y, w, .001)[1])), 1e-12)
        self.assertLessEqual(trace[-1]["objective"], trace[0]["objective"] + 2e-15)
        self.assertTrue(all(point["line_trials"] for point in trace[:-1]))

    def test_held_feature_and_target_changes_cannot_change_training(self):
        groups = [f"group{i:04}" for i in range(1000)]
        x = np.column_stack((np.sin(np.arange(1000)), np.cos(np.arange(1000) / 7)))
        y = np.arange(1000) % 2
        frame = pd.DataFrame(
            {"source_row": range(1000), "group_id": groups, "target": y}
        )
        held = model.partitions(groups, 101) == "test"
        altered = x.copy()
        altered[held] = 999999
        altered_y = y.copy()
        altered_y[held] = 1 - altered_y[held]
        outputs = []
        for xx, yy in [(x, y), (altered, altered_y)]:
            with tempfile.TemporaryDirectory() as temporary, patch.object(
                experiment, "panels", return_value=iter([({"id": 1}, frame, xx, yy)])
            ), patch.object(experiment, "SEEDS", [101]), patch.object(
                experiment, "ALPHAS", [0.1]
            ), patch.object(
                experiment, "PRIORS", [0.01]
            ):
                root = Path(temporary)
                experiment.train(root)
                result = json.loads((root / "results/models/1_101.json").read_bytes())

                def strip_times(value):
                    if isinstance(value, dict):
                        return {
                            k: strip_times(v)
                            for k, v in value.items()
                            if k != "seconds"
                        }
                    return (
                        [strip_times(v) for v in value]
                        if isinstance(value, list)
                        else value
                    )

                outputs.append(strip_times(result))
        self.assertEqual(outputs[0], outputs[1])

    def test_bootstrap_retains_missing_class_draws(self):
        frame = pd.DataFrame({"group_id": ["negative", "positive"], "target": [0, 1]})
        result = experiment.bootstrap(
            frame, np.array([0.0, 0.0]), np.array([1.0, 1.0]), 123
        )
        self.assertTrue(any(not d["valid"] for d in result["draws"]))
        self.assertIsNone(result["interval_95"])
        self.assertEqual(result["interval_status"], "UNVERIFIED_INVALID_CLASS_DRAW")


if __name__ == "__main__":
    unittest.main()
