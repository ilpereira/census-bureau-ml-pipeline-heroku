# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import logging as log
from starter.starter.ml.data import process_data, load_data, save_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference, save_model

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent

if __name__ == "__main__":
    log.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Load in the data.
    data = load_data("census_clean.csv")

    train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
    save_data(train_data, "train_data.csv")
    save_data(test_data, "test_data.csv")

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
        train_data, categorical_features=cat_features, label="salary", training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)
    save_model(model=model, file_name="logreg_model.pkl")
    save_model(model=encoder, file_name="encoder.pkl")
    save_model(model=lb, file_name="label_binarizer.pkl")
    save_model(model=scaler, file_name="scaler.pkl")

    # Compute model metrics on train data
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    log.info(f"Model metrics on train data: precision {precision}, recall: {recall}, fbeta: {fbeta}")
