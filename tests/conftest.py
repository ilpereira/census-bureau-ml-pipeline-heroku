import pandas as pd
import os
from pathlib import Path
from pytest import fixture
from starter.starter.ml.data import process_data

PROJECT_ROOT_DIR = Path(__file__).parent.parent


@fixture(scope='function')
def data():
    df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "starter/data/census_clean.csv"))[:10]

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, _, _, _ = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )

    return X_train, y_train