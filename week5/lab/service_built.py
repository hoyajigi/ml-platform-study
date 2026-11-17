"""빌드 고정 방식: store 에 있는 모델을 BentoModel 로 선언 → bentoml build 시 bento 안에 패킹된다."""
import bentoml
import pandas as pd


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 10}, workers=1)
class IncomeBuilt:
    bento_model = bentoml.models.BentoModel("income-rf-built:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, records: list[dict]) -> list:
        return self.model.predict(pd.DataFrame.from_records(records)).tolist()
