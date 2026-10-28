"""Independent standard-library ARFF reconciliation, without generation imports."""

from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import sqlite3
import struct
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = []
    for record in json.loads((ROOT / "data/manifest.json").read_bytes()):
        raw = ROOT / f"data/raw/{record['id']}.arff"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["raw_sha256"]
        assert (
            hashlib.sha256(
                (ROOT / f"data/metadata/{record['id']}.json").read_bytes()
            ).hexdigest()
            == record["metadata_sha256"]
        )
        assert (
            hashlib.sha256((ROOT / record["panel_file"]).read_bytes()).hexdigest()
            == record["panel_sha256"]
        )
        seen = set()
        rows = []
        source_rows = 0
        attributes = []
        with raw.open(encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("@attribute"):
                    attribute = line.strip().split(None, 2)[1].strip("'\"")
                    attributes.append(attribute)
                if line.strip().lower() == "@data":
                    break
            target_index = attributes.index(record["original_label"])
            indices = [attributes.index(c) for c in record["features"]]
            for row in csv.reader(handle):
                if not row or row[0].startswith("%"):
                    continue
                source_row = source_rows
                source_rows += 1
                label = row[target_index].strip().strip("'\"")
                assert label in {record["positive_label"], record["negative_label"]}
                target = int(label == record["positive_label"])
                values = tuple(float(row[i]) for i in indices)
                assert all(math.isfinite(x) for x in values)
                key = values + (target,)
                if key in seen:
                    continue
                seen.add(key)
                signature = hashlib.sha256(
                    struct.pack(
                        "<" + "d" * len(values), *(0.0 if v == 0 else v for v in values)
                    )
                ).hexdigest()
                rows.append((signature, source_row, label, target, values))
        rows.sort(key=lambda r: (r[0], r[1]))
        group_sizes = {}
        for row in rows:
            group_sizes[row[0]] = group_sizes.get(row[0], 0) + 1
        selected_groups = set()
        count = 0
        for group, size in sorted(group_sizes.items()):
            count += size
            if count <= 8000:
                selected_groups.add(group)
        expected = [row for row in rows if row[0] in selected_groups]
        panel = pd.read_csv(
            ROOT / record["panel_file"],
            float_precision="round_trip",
            dtype={"source_label": str},
        )
        assert len(panel) == len(expected) == record["quality"]["selected_rows"]
        for actual, source in zip(panel.itertuples(index=False, name=None), expected):
            group, source_row, label, target, values = source
            assert tuple(actual[:3]) == (source_row, label, group)
            assert tuple(actual[3:-1]) == values and actual[-1] == target
        assert source_rows == record["quality"]["source_rows"]
        assert (
            source_rows - len(seen)
            == record["quality"]["exact_full_row_value_duplicates_removed"]
        )
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE panel(group_id TEXT, source_row INTEGER, target INTEGER)"
        )
        connection.executemany(
            "INSERT INTO panel VALUES(?,?,?)", [(r[0], r[1], r[3]) for r in expected]
        )
        counts = dict(
            connection.execute(
                "SELECT target,COUNT(*) FROM panel GROUP BY target"
            ).fetchall()
        )
        assert {str(k): v for k, v in counts.items()} == record["quality"][
            "selected_class_counts"
        ]
        split_counts = []
        for seed in [101, 202, 303, 404, 505]:
            connection.create_function(
                "bucket",
                1,
                lambda group: int(
                    hashlib.sha256((group + ":" + str(seed)).encode()).hexdigest()[:8],
                    16,
                )
                % 20,
            )
            result = connection.execute(
                "SELECT CASE WHEN bucket(group_id)<10 THEN 'fit' WHEN bucket(group_id)<13 THEN 'validation' WHEN bucket(group_id)<16 THEN 'calibration' ELSE 'test' END part,target,COUNT(*) FROM panel GROUP BY part,target ORDER BY part,target"
            ).fetchall()
            assert len(result) == 8 and min(row[2] for row in result) >= 20
            split_counts.append({"seed": seed, "partition_class_counts": result})
        evidence.append(
            {
                "dataset": record["id"],
                "all_source_rows_parsed": source_rows,
                "all_selected_values_exact": True,
                "selected_rows": len(expected),
                "whole_groups": len(selected_groups),
                "sql_counts": split_counts,
            }
        )
        print(
            f"Independent source/SQL check PASS: {record['id']} ({source_rows} raw rows)",
            flush=True,
        )
    out = {
        "status": "PASS",
        "method": "CSV/struct parsing and SQLite accounting independent of scipy ingestion",
        "evidence": evidence,
    }
    (ROOT / "results/INDEPENDENT_SOURCE_CHECK.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
