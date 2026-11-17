"""노션 10.6 방식: 서비스 __init__ 에서 MLflow alias 를 해석하고 bentoml.mlflow.import_model 로 가져온다.
(노션은 runs:/{run_id}/model 을 썼지만 MLflow 3 의 logged model 은 models:/ URI 로 참조)"""
import os

import bentoml
import mlflow
import pandas as pd
from mlflow import MlflowClient

NAME = os.environ.get("MODEL_NAME", "income-rf")
ALIAS = os.environ.get("MODEL_ALIAS", "champion")


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 10}, workers=1)
class IncomeLive:
    def __init__(self):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mv = MlflowClient().get_model_version_by_alias(NAME, ALIAS)
        self.info = {"name": NAME, "alias": ALIAS, "version": mv.version, "run_id": mv.run_id}
        bentoml.mlflow.import_model("income-rf-live", f"models:/{NAME}@{ALIAS}")
        bento_model = bentoml.mlflow.get("income-rf-live:latest")
        self.store_tag = str(bento_model.tag)
        self.model = mlflow.pyfunc.load_model(bento_model.path_of(bentoml.mlflow.MLFLOW_MODEL_FOLDER))

    def _proba(self, records):
        return self.model.predict(pd.DataFrame.from_records(records)).tolist()

    @bentoml.api
    def predict(self, records: list[dict]) -> list:
        return self._proba(records)

    @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=20)
    def predict_batched(self, records: list[dict]) -> list:
        return self._proba(records)

    @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=1000)
    def predict_batched_1s(self, records: list[dict]) -> list:
        return self._proba(records)

    @bentoml.api
    def predict_tagged(self, records: list[dict], ctx: bentoml.Context) -> list:
        """기본 동작이 아닌 대조군: 응답 헤더에 버전을 직접 실어 보낸다."""
        ctx.response.headers["x-model-version"] = self.info["version"]
        ctx.response.headers["x-model-run-id"] = self.info["run_id"]
        ctx.response.headers["x-bento-model"] = self.store_tag
        return self._proba(records)
