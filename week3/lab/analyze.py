"""Summarize drift_lab jsonl outputs into CSV tables (results/) and print markdown.

Usage: python analyze.py --in <out-dir> --tags 050 latest --data adult.data --results results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import distance

from drift_lab import CAT_COLS, NUM_COLS, SCENARIOS, load_split

ALL_COLS = CAT_COLS + NUM_COLS


def read_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if "meta" not in r and "meta_end" not in r]


def read_meta(path: str) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "meta" in r or "meta_end" in r:
                    meta.update(r)
    return meta


def grid_table(rows: List[dict], tag: str) -> pd.DataFrame:
    recs = []
    for r in rows:
        rec = {"tag": tag, "config": r["config"], "scenario": r["scenario"], "n": r["n"],
               "ref_size": r["ref_size"], "cat_share": r["cat_share"],
               "dataset_drift_cat8": r["dataset_drift_cat8"]}
        for c in ALL_COLS:
            rec[f"drift_{c}"] = r["cols"][c][2]
            rec[f"test_{c}"] = r["cols"][c][0]
            rec[f"score_{c}"] = r["cols"][c][1]
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    keys = ["tag", "config", "scenario", "ref_size", "n"]
    agg = df.groupby(keys).agg(
        reps=("cat_share", "size"),
        mean_cat_share=("cat_share", "mean"),
        dataset_drift_rate=("dataset_drift_cat8", "mean"),
        **{f"rate_{c}": (f"drift_{c}", "mean") for c in ALL_COLS},
        **{f"median_score_{c}": (f"score_{c}", "median") for c in ALL_COLS},
    ).reset_index()
    agg["rate_any_other7cat"] = df.assign(
        any_other=df[[f"drift_{c}" for c in CAT_COLS if c != "Sex"]].any(axis=1)
    ).groupby(keys)["any_other"].mean().values
    test_labels = []
    for _, g in df.groupby(keys):
        test_labels.append(";".join(
            f"{c}={'/'.join(sorted(set(g[f'test_{c}'])))}" for c in ["Sex", "Education", "Age"]))
    agg["tests"] = test_labels
    return agg


def sliding_table(rows: List[dict], tag: str) -> pd.DataFrame:
    recs = []
    for r in rows:
        v = np.array(r["verdicts"], dtype=bool)
        longest, cur = 0, 0
        for x in v:
            cur = cur + 1 if x else 0
            longest = max(longest, cur)
        first = int(np.argmax(v)) if v.any() else -1
        # number of alarm episodes (False->True transitions)
        episodes = int(v[0]) + int(np.sum(~v[:-1] & v[1:]))
        recs.append({"tag": tag, "window": r["window"], "stream": r["stream"],
                     "n_reports": r["n_reports"], "flag_rate": v.mean(),
                     "mean_share": float(np.mean(r["shares"])),
                     "first_alarm_report": first, "longest_run": longest,
                     "alarm_episodes": episodes,
                     **{f"col_rate_{c}": r["col_hits"][c] / r["n_reports"] for c in CAT_COLS}})
    return pd.DataFrame(recs).sort_values(["window", "stream"])


def population_effects(data: str) -> pd.DataFrame:
    """Score each scenario would get with infinite current data (JS for Sex, normed WD for Age)."""
    ref, _ = load_split(data)
    p0 = float((ref["Sex"] == "Female").mean())
    sd = float(np.std(ref["Age"]))
    recs = []
    for name, (pp, age) in SCENARIOS.items():
        js = float(distance.jensenshannon([p0, 1 - p0], [p0 + pp, 1 - p0 - pp]))
        recs.append({"scenario": name, "sex_pp": pp, "female_ref": round(p0, 4),
                     "js_population": round(js, 4), "age_add": age,
                     "age_std_ref": round(sd, 3), "wd_norm_population": round(age / sd, 4)})
    return pd.DataFrame(recs)


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.results, exist_ok=True)

    grids, refs, slides, vals, metas = [], [], [], [], []
    for tag in args.tags:
        p = lambda e: os.path.join(args.indir, f"{e}-{tag}.jsonl")  # noqa: E731
        grids.append(grid_table(read_rows(p("grid")), tag))
        refs.append(grid_table(read_rows(p("refsize")), tag))
        slides.append(sliding_table(read_rows(p("sliding")), tag))
        v = pd.DataFrame(read_rows(p("validate")))
        if not v.empty:
            vals.append({"tag": tag, "trials": len(v),
                         "flag_match": float((v.evidently_dataset_drift == v.derived_dataset_drift).mean()),
                         "share_match": float((v.evidently_share == v.derived_share).mean())})
        for e in ("validate", "grid", "refsize", "sliding"):
            m = read_meta(p(e))
            if m:
                metas.append({"tag": tag, "exp": e, **m})

    out = {
        "grid_summary.csv": pd.concat(grids),
        "refsize_summary.csv": pd.concat(refs),
        "sliding_summary.csv": pd.concat(slides),
        "validate_summary.csv": pd.DataFrame(vals),
        "population_effects.csv": population_effects(args.data),
        "run_meta.csv": pd.DataFrame(metas),
    }
    for name, df in out.items():
        df.to_csv(os.path.join(args.results, name), index=False, float_format="%.4f")
        print(f"\n## {name}\n")
        print(df.to_string(index=False, max_colwidth=60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
