"""alias champion 을 지정한 버전으로 옮긴다. 사용: python move_alias.py <version>"""
import os, sys
from mlflow import MlflowClient
c = MlflowClient()
c.set_registered_model_alias(os.environ["MODEL_NAME"], "champion", sys.argv[1])
print("champion ->", c.get_model_version_by_alias(os.environ["MODEL_NAME"], "champion").version)
