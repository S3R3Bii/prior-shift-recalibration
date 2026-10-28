"""Freeze exact binary-label panels for controlled prior-shift scenarios."""

from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    44125: {
        "positive": "g",
        "negative": "h",
        "kind": "Monte Carlo simulation",
        "creator": "R. Bock",
        "doi": "10.24432/C52C8B",
        "rights_url": "https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
    },
    44128: {
        "positive": "True",
        "negative": "False",
        "kind": "Public event benchmark derived from MiniBooNE experiment",
        "creator": "Byron Roe",
        "doi": "10.24432/C5QC87",
        "rights_url": "https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification",
    },
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def signature(values):
    values = np.asarray(values, dtype="<f8").copy()
    values[values == 0] = 0.0
    values[np.isnan(values)] = np.nan
    return sha(values.tobytes())


def panel_from_raw(raw, description, source):
    values, _ = arff.loadarff(io.StringIO(raw.decode("utf-8")))
    frame = pd.DataFrame(values)
    label = description["default_target_attribute"]
    labels = frame[label].map(
        lambda v: v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
    )
    assert set(labels) == {source["positive"], source["negative"]}
    ignored = description.get("ignore_attribute", [])
    if isinstance(ignored, str):
        ignored = [ignored]
    ignored = list(ignored or [])
    if description.get("row_id_attribute"):
        ignored.append(description["row_id_attribute"])
    features = [c for c in frame if c != label and c not in ignored]
    numeric = frame[features].apply(pd.to_numeric, errors="raise")
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
    numeric.insert(0, "source_row", np.arange(len(frame)))
    numeric.insert(1, "source_label", labels)
    numeric["target"] = (labels == source["positive"]).astype(int)
    unique = numeric.drop_duplicates(subset=features + ["target"], keep="first").copy()
    unique.insert(
        2,
        "group_id",
        [signature(row) for row in unique[features].to_numpy(dtype=float)],
    )
    sizes = unique.groupby("group_id", sort=True).size()
    groups = sizes.loc[sizes.cumsum() <= 8000].index
    panel = (
        unique.loc[unique.group_id.isin(groups)]
        .sort_values(["group_id", "source_row"])
        .reset_index(drop=True)
    )
    quality = {
        "source_rows": len(frame),
        "exact_full_row_value_duplicates_removed": len(frame) - len(unique),
        "distinct_full_rows": len(unique),
        "selected_rows": len(panel),
        "selected_feature_groups": panel.group_id.nunique(),
        "selected_class_counts": {
            str(k): int(v) for k, v in panel.target.value_counts().items()
        },
        "conflicting_label_feature_groups": int(
            (panel.groupby("group_id").target.nunique() > 1).sum()
        ),
        "feature_count": len(features),
        "selected_values_below_minus900": int((panel[features] < -900).sum().sum()),
        "extreme_value_policy": "Retained as supplied; no unverified sentinel interpretation or outcome-based cleaning",
    }
    return panel, features, quality


def main():
    records = []
    for number, source in SOURCES.items():
        meta = ROOT / f"data/metadata/{number}.json"
        description = json.loads(meta.read_bytes())["data_set_description"]
        assert description["status"] == "active"
        path = ROOT / f"data/raw/{number}.arff"
        if path.exists():
            raw = path.read_bytes()
            retrieved = "Exact local source download reused"
        else:
            raw = urllib.request.urlopen(description["url"], timeout=120).read()
            path.write_bytes(raw)
            retrieved = datetime.now(timezone.utc).isoformat()
        panel, features, quality = panel_from_raw(raw, description, source)
        out = ROOT / f"data/panels/{number}.csv.gz"
        out.write_bytes(
            gzip.compress(
                panel.to_csv(
                    index=False, float_format="%.17g", lineterminator="\n"
                ).encode("utf-8"),
                mtime=0,
            )
        )
        records.append(
            {
                "id": number,
                "name": description["name"],
                "version": description["version"],
                "original_label": description["default_target_attribute"],
                "positive_label": source["positive"],
                "negative_label": source["negative"],
                "features": features,
                "provenance_kind": source["kind"],
                "creator": source["creator"],
                "doi": source["doi"],
                "openml_license_field": description["licence"],
                "original_rights_boundary": "CC BY4.0 with creator/source attribution; changes declared",
                "rights_url": source["rights_url"],
                "source_url": description["url"],
                "retrieved_utc": retrieved,
                "raw_sha256": sha(raw),
                "raw_bytes": len(raw),
                "metadata_sha256": sha(meta.read_bytes()),
                "panel_file": out.relative_to(ROOT).as_posix(),
                "panel_sha256": sha(out.read_bytes()),
                "quality": quality,
            }
        )
        print(
            json.dumps({"dataset": number, "name": description["name"], **quality}),
            flush=True,
        )
    (ROOT / "data/manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "results/source_quality.json").write_text(
        json.dumps(
            {
                "status": "SOURCE_PANELS_READY_FOR_INDEPENDENT_RECONCILIATION",
                "datasets": {str(r["id"]): r["quality"] for r in records},
                "scope": "Controlled benchmark derivatives; no deployment prevalence or new physical measurement",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
