"""v1, v2 학습 → 같은 이름으로 레지스트리 등록 → alias champion=v1.
pyfunc 예측이 확률(predict_proba)을 내도록 pyfunc_predict_fn 지정."""
import json
import os
import sys

import mlflow
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load, probe_rows

NAME = os.environ["MODEL_NAME"]
CONFIGS = {
    "v1": dict(n_estimators=50, max_depth=8, random_state=1),
    "v2": dict(n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=2),
}


def main():
    x, y = load()
    print("rows", len(x), "pos_rate", round(y.mean(), 4))
    mlflow.set_experiment("week5-serving")
    sample = probe_rows(5)
    out = {}
    for tag, params in CONFIGS.items():
        pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                             RandomForestClassifier(**params))
        pipe.fit(x, y)
        with mlflow.start_run(run_name=tag) as run:
            mlflow.log_params(params)
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=sample,
                                            pyfunc_predict_fn="predict_proba",
                                            skops_trusted_types=["sklearn.tree._tree.Tree"],
                                            registered_model_name=NAME)
        out[tag] = {"run_id": run.info.run_id, "version": info.registered_model_version,
                    "model_id": getattr(info, "model_id", None), "params": params}
        print(tag, out[tag])
    MlflowClient().set_registered_model_alias(NAME, "champion", out["v1"]["version"])
    json.dump(out, open(os.path.join(os.environ["LAB"], "versions.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
