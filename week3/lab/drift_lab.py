"""Evidently data-drift verdict vs. current-window size and true shift size.

Reproduces the book's 6.2.3 setup (Adult income data, 8 categorical columns,
reference = training data, current = window of N requests) and measures:

  grid     : false-alarm rate (no shift) and detection rate (shifted) over n
  refsize  : reference size 1000 vs 1001 (where Evidently switches default test)
  sliding  : book-style sliding window, one report per request
  validate : derived dataset flag == Evidently's own DatasetDriftMetric flag

Works with evidently 0.5.0 (book) and 0.7.x (uses evidently.legacy, which the
new-API DataDriftPreset/DriftedColumnsCount wraps).

Usage:
  python drift_lab.py --data adult.data --exp grid --reps 100 --out grid.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # evidently >= 0.7: old API lives under evidently.legacy
    from evidently.legacy.metrics import DataDriftTable, DatasetDriftMetric
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report
except ImportError:  # evidently 0.5.0 (book)
    from evidently import ColumnMapping
    from evidently.metrics import DataDriftTable, DatasetDriftMetric
    from evidently.report import Report

import evidently

ADULT_COLUMNS = (
    "Age Workclass fnlwgt Education Education-Num Marital-Status Occupation "
    "Relationship Race Sex Capital-gain Capital-loss Hours-per-week "
    "Native_country income"
).split()
# Same 8 columns as the book's evidently_reports.py / FEATURE_LIST
CAT_COLS = [
    "Education", "Marital-Status", "Native_country", "Occupation",
    "Race", "Relationship", "Sex", "Workclass",
]
NUM_COLS = ["Age", "Hours-per-week"]  # extra, not in the book's report
N_GRID = [30, 50, 100, 200, 500, 1000, 1001, 2000, 5000]
# scenario -> (Sex: +pp Female share, Age: +years). Both shifts are applied in
# the same trial; Age is not one of the 8 book columns, so the 8-column dataset
# verdict only "sees" the Sex shift.
SCENARIOS: Dict[str, Tuple[float, float]] = {
    "null": (0.0, 0.0),
    "s1": (0.05, 1.0),
    "s2": (0.10, 2.0),
    "s3": (0.20, 4.0),
}
CONFIGS = {
    "default": {},
    "pvalue": {"cat_stattest": "chisquare", "num_stattest": "ks",
               "cat_stattest_threshold": 0.05, "num_stattest_threshold": 0.05},
}
DRIFT_SHARE = 0.5  # evidently default (options/data_drift.py drift_share)

_STATE: Dict[str, pd.DataFrame] = {}


def load_split(path: str, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reference = 70% of adult.data (training-set stand-in), pool = other 30%."""
    df = pd.read_csv(path, header=None, names=ADULT_COLUMNS, skipinitialspace=True)
    df = df.dropna(subset=CAT_COLS + NUM_COLS)
    ref = df.sample(frac=0.7, random_state=seed)
    pool = df.drop(ref.index)
    cols = CAT_COLS + NUM_COLS
    return ref[cols].reset_index(drop=True), pool[cols].reset_index(drop=True)


def _init(path: str) -> None:
    ref, pool = load_split(path)
    _STATE["ref"], _STATE["pool"] = ref, pool


def apply_shift(cur: pd.DataFrame, sex_pp: float, age_add: float,
                p_female: float, rng: np.random.Generator) -> pd.DataFrame:
    """Return a shifted copy. Sex: relabel Male->Female with prob q so the expected
    Female share becomes p_female + sex_pp (marginal-only change, other columns
    untouched). Age: add a constant (mean shift, shape unchanged)."""
    out = cur.copy()
    if sex_pp > 0:
        q = sex_pp / (1.0 - p_female)
        male = out["Sex"].to_numpy() == "Male"
        flip = male & (rng.random(len(out)) < q)
        out["Sex"] = np.where(flip, "Female", out["Sex"].to_numpy())
    if age_add:
        out["Age"] = out["Age"] + age_add
    return out


def run_report(ref: pd.DataFrame, cur: pd.DataFrame, config: str) -> Dict[str, list]:
    """Per-column [stattest_name, score, drifted] from Evidently's DataDriftTable."""
    cm = ColumnMapping(categorical_features=CAT_COLS, numerical_features=NUM_COLS,
                       target=None, prediction=None)
    report = Report(metrics=[DataDriftTable(columns=CAT_COLS + NUM_COLS, **CONFIGS[config])])
    report.run(reference_data=ref, current_data=cur, column_mapping=cm)
    by_col = report.as_dict()["metrics"][0]["result"]["drift_by_columns"]
    return {c: [v["stattest_name"], float(v["drift_score"]), bool(v["drift_detected"])]
            for c, v in by_col.items()}


def summarize(cols: Dict[str, list]) -> Dict[str, float]:
    """Dataset verdict for the book's 8 categorical columns: share >= 0.5
    (evidently calculations/data_drift.py get_dataset_drift)."""
    n_drift = sum(cols[c][2] for c in CAT_COLS)
    share = n_drift / len(CAT_COLS)
    return {"cat_drifted": n_drift, "cat_share": share,
            "dataset_drift_cat8": share >= DRIFT_SHARE}


def _grid_job(job: Tuple[str, str, int, int, int, Optional[int]]) -> dict:
    config, scenario, n, rep, seed, ref_size = job
    rng = np.random.default_rng(seed)
    ref, pool = _STATE["ref"], _STATE["pool"]
    if ref_size is not None:
        ref = ref.sample(ref_size, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    cur = pool.sample(n, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    p_female = float((_STATE["ref"]["Sex"] == "Female").mean())
    sex_pp, age_add = SCENARIOS[scenario]
    cur = apply_shift(cur, sex_pp, age_add, p_female, rng)
    cols = run_report(ref, cur, config)
    return {"config": config, "scenario": scenario, "n": n, "rep": rep,
            "ref_size": len(ref), "cols": cols, **summarize(cols)}


def _sliding_job(job: Tuple[int, int, int, int]) -> dict:
    """Book MonitoringService: append 1 row per request, keep last N rows,
    run report on every request once the window is full."""
    window, stream_id, n_reports, seed = job
    rng = np.random.default_rng(seed)
    ref, pool = _STATE["ref"], _STATE["pool"]
    stream = pool.sample(window + n_reports - 1,
                         random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    verdicts, shares, col_hits = [], [], {c: 0 for c in CAT_COLS}
    for i in range(n_reports):
        cur = stream.iloc[i:i + window].reset_index(drop=True)
        cols = run_report(ref, cur, "default")
        s = summarize(cols)
        verdicts.append(bool(s["dataset_drift_cat8"]))
        shares.append(s["cat_share"])
        for c in CAT_COLS:
            col_hits[c] += int(cols[c][2])
    return {"window": window, "stream": stream_id, "n_reports": n_reports,
            "verdicts": verdicts, "shares": shares, "col_hits": col_hits}


def _validate_job(job: Tuple[int, int]) -> dict:
    """Check the derived 8-column flag against Evidently's own DatasetDriftMetric."""
    n, seed = job
    rng = np.random.default_rng(seed)
    ref, pool = _STATE["ref"], _STATE["pool"]
    cur = pool.sample(n, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    cm = ColumnMapping(categorical_features=CAT_COLS, numerical_features=[],
                       target=None, prediction=None)
    report = Report(metrics=[DatasetDriftMetric(columns=CAT_COLS)])
    report.run(reference_data=ref[CAT_COLS], current_data=cur[CAT_COLS], column_mapping=cm)
    res = report.as_dict()["metrics"][0]["result"]
    derived = summarize(run_report(ref, cur, "default"))
    return {"n": n, "seed": seed, "evidently_dataset_drift": bool(res["dataset_drift"]),
            "evidently_share": float(res["share_of_drifted_columns"]),
            "derived_dataset_drift": bool(derived["dataset_drift_cat8"]),
            "derived_share": derived["cat_share"]}


def build_jobs(exp: str, reps: int) -> Tuple[object, list]:
    seed = 1000
    jobs: list = []
    if exp == "grid":
        for config in CONFIGS:
            for scenario in SCENARIOS:
                for n in N_GRID:
                    for r in range(reps):
                        seed += 1
                        jobs.append((config, scenario, n, r, seed, None))
        return _grid_job, jobs
    if exp == "refsize":
        for ref_size in (1000, 1001):
            for scenario in ("null", "s2", "s3"):
                for n in (50, 100, 500, 1000):
                    for r in range(reps):
                        seed += 1
                        jobs.append(("default", scenario, n, r, seed, ref_size))
        return _grid_job, jobs
    if exp == "sliding":
        for window in (50, 100, 200, 500):
            for s in range(reps):
                seed += 1
                jobs.append((window, s, 200, seed))
        return _sliding_job, jobs
    if exp == "validate":
        for n in (50, 100, 1000, 5000):
            for r in range(reps):
                seed += 1
                jobs.append((n, seed))
        return _validate_job, jobs
    raise ValueError(f"unknown exp {exp}")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--exp", required=True, choices=["grid", "refsize", "sliding", "validate"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    func, jobs = build_jobs(args.exp, args.reps)
    t0 = time.time()
    meta = {"meta": True, "exp": args.exp, "reps": args.reps,
            "evidently": evidently.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "python": sys.version.split()[0], "jobs": len(jobs)}
    with open(args.out, "w") as f, Pool(args.workers, initializer=_init, initargs=(args.data,)) as pool:
        f.write(json.dumps(meta) + "\n")
        for row in pool.imap_unordered(func, jobs, chunksize=4):
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps({"meta_end": True, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    print(f"{args.exp}: {len(jobs)} jobs, {time.time() - t0:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
