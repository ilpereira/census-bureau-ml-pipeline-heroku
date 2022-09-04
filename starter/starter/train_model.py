# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from pathlib import Path

import sys
import logging as log

# Add the necessary imports for the starter code.
from starter.starter.ml.data import process_data, load_data
from starter.starter.ml.model import train_model, compute_model_metrics_slices, compute_model_metrics, inference, save_model

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent

if __name__ == "__main__":
    log.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Add code to load in the data.
    data = load_data("census_clean.csv")

    # Optional enhancement,use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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

    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb, scaler = process_data(test, categorical_features=cat_features, label='salary',
                                                       training=False,
                                                       encoder=encoder, lb=lb, scaler=scaler)

    # Train and save a model.
    model = train_model(X_train, y_train)
    save_model(model=model, file_name="logreg_model.pkl")
    save_model(model=encoder, file_name="encoder.pkl")
    save_model(model=lb, file_name="label_binarizer.pkl")
    save_model(model=scaler, file_name="scaler.pkl")

    # Compute model metrics
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    log.info(f"Model metrics on train data: precision {precision}, recall: {recall}, fbeta: {fbeta}")

    # TODO: Compute model metrics on slices (???)
    # compute_model_metrics_slices(model=model, test_data=test, categorical_features=cat_features, encoder=encoder, lb=lb)
