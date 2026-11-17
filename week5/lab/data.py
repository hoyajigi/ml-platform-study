"""Adult 데이터 로드 (2주차와 같은 범주형 8개 입력, '?' 있는 행 제거 → 30,162행)."""
import os

import pandas as pd

COLS = ["Age", "Workclass", "fnlwgt", "Education", "Education_num", "Marital-Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital_gain", "Capital_loss",
        "Hours_per_week", "Native_country", "Target"]
FEATURES = ["Workclass", "Education", "Marital-Status", "Occupation",
            "Relationship", "Race", "Sex", "Native_country"]


def load():
    df = pd.read_csv(os.environ["ADULT_CSV"], header=None, names=COLS,
                     skipinitialspace=True, na_values="?").dropna()
    x = df[FEATURES].reset_index(drop=True)
    y = (df["Target"] == ">50K").astype(int).reset_index(drop=True)
    return x, y


def probe_rows(n=1000):
    """서빙 비교용 고정 입력 n행 (random_state 고정)."""
    x, _ = load()
    return x.sample(n=n, random_state=7).reset_index(drop=True)
