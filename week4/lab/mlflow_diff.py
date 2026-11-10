"""MLflow(sqlite) 에 남은 run 쌍을 비교: params 중 다른 키, 마지막 metric 값 차이, 태그(코드 버전 등) 차이."""
import argparse
import json

from mlflow.tracking import MlflowClient


def by_name(client, exp_name):
    exp = client.get_experiment_by_name(exp_name)
    return {r.info.run_name: r for r in client.search_runs([exp.experiment_id], max_results=500)}


def diff(a, b):
    pa, pb = a.data.params, b.data.params
    keys = sorted(set(pa) | set(pb))
    p_diff = {k: (pa.get(k), pb.get(k)) for k in keys if pa.get(k) != pb.get(k)}
    ma, mb = a.data.metrics, b.data.metrics
    m_diff = {k: (ma.get(k), mb.get(k)) for k in sorted(set(ma) | set(mb)) if ma.get(k) != mb.get(k)}
    ta = {k: v for k, v in a.data.tags.items() if not k.startswith("mlflow.runName")}
    tb = {k: v for k, v in b.data.tags.items() if not k.startswith("mlflow.runName")}
    t_diff = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
    return {"n_params": len(keys), "params_differing": p_diff, "n_metrics": len(set(ma) | set(mb)),
            "metrics_differing": m_diff, "tags_differing": t_diff,
            "source_tags": {k: ta.get(k) for k in ("mlflow.source.name", "mlflow.source.git.commit")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="experiment:runA:runB")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    client = MlflowClient(tracking_uri=args.uri)
    res = []
    for spec in args.pairs:
        exp, a, b = spec.split(":")
        runs = by_name(client, exp)
        res.append({"experiment": exp, "a": a, "b": b, **diff(runs[a], runs[b])})
    with open(args.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
