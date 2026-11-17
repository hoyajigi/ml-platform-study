"""빌드 고정 방식용: 지금 alias 가 가리키는 모델을 BentoML store 로 한 번 가져온다."""
import os
import bentoml

name, alias = os.environ["MODEL_NAME"], "champion"
m = bentoml.mlflow.import_model("income-rf-built", f"models:/{name}@{alias}")
print("imported", m.tag, m.path)
