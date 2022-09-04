from starter.starter.ml.model import inference, train_model, compute_model_metrics
from sklearn.linear_model import LogisticRegression
import numpy as np


def test_inference(data):
    X_train, y_train = data

    model = train_model(X_train=X_train, y_train=y_train)

    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray)
    assert len(X_train) == len(preds)


def test_train_model(data):
    X_train, y_train = data

    model = train_model(X_train=X_train, y_train=y_train)
    assert isinstance(model, LogisticRegression)


def test_compute_model_metrics(data):
    X_train, y_train = data

    model = train_model(X_train=X_train, y_train=y_train)
    preds = model.predict(X_train)

    precision, recall, fbeta = compute_model_metrics(y=y_train, preds=preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
