"""evidently >= 0.7 only: check that the new API (Report + DataDriftPreset) gives the
same drifted-column count/share and per-column scores as the legacy DataDriftTable
path used by drift_lab.py, on identical samples (8 book columns).

Usage: python newapi_check.py --data adult.data --reps 20 --out newapi.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from drift_lab import CAT_COLS, load_split, run_report


def new_api(ref, cur) -> dict:
    dd = DataDefinition(categorical_columns=CAT_COLS)
    snap = Report([DataDriftPreset(columns=CAT_COLS)]).run(
        Dataset.from_pandas(cur[CAT_COLS], data_definition=dd),
        Dataset.from_pandas(ref[CAT_COLS], data_definition=dd),
    )
    out = {"count": None, "share": None, "scores": {}}
    for m in snap.dict()["metrics"]:
        name, value = m["metric_name"], m["value"]
        if name.startswith("DriftedColumnsCount"):
            out["count"], out["share"] = value["count"], value["share"]
        elif name.startswith("ValueDrift"):
            col = name.split("column=")[1].split(",")[0].rstrip(")")
            out["scores"][col] = float(value)
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ref, pool = load_split(args.data)
    rng = np.random.default_rng(7)
    mismatches = 0
    with open(args.out, "w") as f:
        for n in (50, 100, 1000):
            for r in range(args.reps):
                cur = pool.sample(n, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
                legacy = run_report(ref, cur, "default")
                leg_count = sum(legacy[c][2] for c in CAT_COLS)
                new = new_api(ref, cur)
                max_score_diff = max(abs(new["scores"][c] - legacy[c][1]) for c in CAT_COLS)
                ok = new["count"] == leg_count and max_score_diff < 1e-9
                mismatches += int(not ok)
                f.write(json.dumps({"n": n, "rep": r, "legacy_count": leg_count,
                                    "new_count": new["count"], "new_share": new["share"],
                                    "max_score_diff": max_score_diff, "match": ok}) + "\n")
    print(f"mismatches: {mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
