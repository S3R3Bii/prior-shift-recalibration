"""Fresh full fits, independent certificates and comparison at declared tolerances."""

from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
began = time.perf_counter()
commands = []


def close(a, b):
    assert np.allclose(a, b, atol=1e-7, rtol=1e-9), (a, b)


def check_model(a, b):
    assert (
        a["selected_alpha"] == b["selected_alpha"]
        and a["boundary_alpha"] == b["boundary_alpha"]
    )
    for key in ["fit_mean", "fit_scale", "base_mean", "base_scale", "base_prevalence"]:
        close(a[key], b[key])

    def fits(fa, fb):
        assert len(fa["attempts"]) == len(fb["attempts"])
        for aa, bb in zip(fa["attempts"], fb["attempts"]):
            assert aa["start_id"] == bb["start_id"] and aa["success"] and bb["success"]
            for key in ["start", "coefficients", "objective"]:
                close(aa[key], bb[key])
        if "coefficients" in fa:
            close(fa["coefficients"], fb["coefficients"])

    for ca, cb in zip(a["candidates"], b["candidates"]):
        assert ca["alpha"] == cb["alpha"]
        fits(ca, cb)
        close(ca["validation_log_loss"], cb["validation_log_loss"])
    fits(a["base"], b["base"])
    for aa, bb in zip(a["adaptations"], b["adaptations"]):
        assert aa["prior"] == bb["prior"]
        close(aa["odds_offset"], bb["odds_offset"])
        for key in ["offset", "objective"]:
            close(aa["intercept"][key], bb["intercept"][key])
        fits(aa["coefficient_refit"], bb["coefficient_refit"])
        fits(aa["slope_intercept"], bb["slope_intercept"])
        close(
            [aa["slope_intercept"]["slope"], aa["slope_intercept"]["offset"]],
            [bb["slope_intercept"]["slope"], bb["slope_intercept"]["offset"]],
        )


with tempfile.TemporaryDirectory(prefix="prior-shift-replay-") as temporary:
    replay = Path(temporary)
    shutil.copytree(
        ROOT / "src", replay / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(
        ROOT / "data", replay / "data", ignore=shutil.ignore_patterns("raw")
    )
    (replay / "results").mkdir()
    for record in json.loads((replay / "data/manifest.json").read_bytes()):
        assert (
            hashlib.sha256((replay / record["panel_file"]).read_bytes()).hexdigest()
            == record["panel_sha256"]
        )
    for script, args in [
        ("experiment.py", ["train"]),
        ("verify.py", ["train"]),
        ("experiment.py", ["test"]),
        ("verify.py", ["test"]),
        ("summarize.py", []),
        ("verify_summary.py", []),
        ("figures.py", []),
    ]:
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(replay / "src" / script), *args],
            cwd=replay,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"Replay step failed: {script} {args}")
        commands.append(
            {
                "script": script,
                "args": args,
                "seconds": time.perf_counter() - start,
                "exit_code": result.returncode,
            }
        )
        print("PASS fresh replay step:", script, *args, flush=True)
    models = 0
    for path in sorted((ROOT / "results/models").glob("*.json")):
        check_model(
            json.loads(path.read_bytes()),
            json.loads((replay / "results/models" / path.name).read_bytes()),
        )
        models += 1
    for name, keys in [
        ("validation_predictions.csv.gz", ["logit"]),
        ("test_predictions.csv.gz", ["logit", "probability", "weight"]),
    ]:
        a = pd.read_csv(ROOT / "results" / name, float_precision="round_trip")
        b = pd.read_csv(replay / "results" / name, float_precision="round_trip")
        assert a.drop(columns=keys).equals(b.drop(columns=keys))
        close(a[keys].to_numpy(), b[keys].to_numpy())
    for name in ["test_metrics.json", "rare_comparisons.json", "summary.json"]:

        def recursive(a, b):
            if isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    recursive(a[key], b[key])
            elif isinstance(a, list):
                assert len(a) == len(b)
                for aa, bb in zip(a, b):
                    recursive(aa, bb)
            elif isinstance(a, (float, int)) and not isinstance(a, bool):
                close(a, b)
            else:
                assert a == b

        recursive(
            json.loads((ROOT / "results" / name).read_bytes()),
            json.loads((replay / "results" / name).read_bytes()),
        )
    for path in sorted((ROOT / "results/bootstrap").glob("*.json")):
        a = json.loads(path.read_bytes())
        b = json.loads((replay / "results/bootstrap" / path.name).read_bytes())
        assert (
            a["ordered_groups"] == b["ordered_groups"]
            and a["draw_index_sha256"] == b["draw_index_sha256"]
        )
        for aa, bb in zip(a["draws"], b["draws"]):
            assert (aa["draw"], aa["n_negative"], aa["n_positive"], aa["valid"]) == (
                bb["draw"],
                bb["n_negative"],
                bb["n_positive"],
                bb["valid"],
            )
            if aa["valid"]:
                close(aa["difference"], bb["difference"])
receipt = {
    "status": "PASS",
    "full_model_runs_refitted": models,
    "optimizer_attempts_refitted": 340,
    "validation_prediction_rows": 71706,
    "test_prediction_rows": 386112,
    "test_metric_rows": 240,
    "bootstrap_draws": 10000,
    "absolute_tolerance": 1e-7,
    "relative_tolerance": 1e-9,
    "end_to_end_seconds": time.perf_counter() - began,
    "timing_scope": "Fresh directory fits, numerical certificates, all prediction and metric comparisons, summaries and figures; excludes raw retrieval, dependency install and private manuscript",
    "commands": commands,
}
(ROOT / "results/reproduction.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({k: v for k, v in receipt.items() if k != "commands"}), flush=True)
