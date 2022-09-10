import logging
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from starter.starter.ml.data import process_data
import pickle
import os
from pathlib import Path

log = logging.getLogger(__name__)
MODEL_PATH = os.path.join(Path(__file__).parent.parent.parent, "model")
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def save_model(model, file_name):
    with open(os.path.join(MODEL_PATH, file_name), "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    with open(os.path.join(MODEL_PATH, file_name), "rb") as f:
        model = pickle.load(f)
        return model


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42, solver="liblinear")
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,recall,and F1.

    Inputs
    ------
    y : np.array
        Known labels,binarized.
    preds : np.array
        Predicted labels,binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def compute_model_metrics_slices(model, test_data: pd.DataFrame, categorical_features: List, encoder, lb, scaler):
    performance_metrics_output_file = os.path.join(PROJECT_ROOT_DIR, 'slice_output.txt')
    with open(performance_metrics_output_file, 'w') as f:
        for feature in categorical_features:
            log.info(f"------------------ Feature '{feature}' ------------------")
            f.write(f"------------------ Feature '{feature}' ------------------\n")
            for feature_value in test_data[feature].unique():
                df = test_data[test_data[feature] == feature_value]
                X_test, y_test, encoder, lb, scaler = process_data(X=df, categorical_features=categorical_features,
                                                                   label='salary',
                                                                   training=False,
                                                                   encoder=encoder, lb=lb, scaler=scaler)
                preds = inference(model, X_test)
                precision, recall, fbeta = compute_model_metrics(y_test, preds)
                log.info(f"Feature value '{feature_value}' -> precision: {precision}, recall: {recall}, fbeta: {fbeta}")
                f.write(f"Feature value '{feature_value}' -> precision: {precision}, recall: {recall}, fbeta: {fbeta}\n")
            f.write("\n")
