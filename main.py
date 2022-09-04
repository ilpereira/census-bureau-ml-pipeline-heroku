# Put the code for your API here.
# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
from pathlib import Path
import os

import sys
import logging as log

# Add the necessary imports for the starter code.
#from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics_slices, compute_model_metrics, inference

PROJECT_ROOT_DIR = Path(__file__).parent

if __name__ == "__main__":
    print(sys.path)

    # log.basicConfig(
    #         format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
    #         level=log.INFO,
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #         stream=sys.stdout,
    #     )
    #
    # # Add code to load in the data.
    # data = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data/census_clean.csv"))
    #
    # # Optional enhancement,use K-fold cross validation instead of a train-test split.
    # train, test = train_test_split(data, test_size=0.20)
    #
    # cat_features = [
    #     "workclass",
    #     "education",
    #     "marital-status",
    #     "occupation",
    #     "relationship",
    #     "race",
    #     "sex",
    #     "native-country",
    # ]
    #
    # X_train, y_train, encoder, lb = process_data(
    #     train, categorical_features=cat_features, label="salary", training=True
    # )
    #
    # # Proces the test data with the process_data function.
    # X_test, y_test, encoder, lb = process_data(test, categorical_features=cat_features, label='salary', training=False,
    #                                            encoder=encoder, lb=lb)
    #
    # # Train and save a model.
    # model = train_model(X_train, y_train)
    # dump(model, "../model/logreg_model.joblib")
    #
    # # Compute model metrics
    # preds = inference(model, X_test)
    # precision, recall, fbeta = compute_model_metrics(y_test, preds)
    # log.info(f"Model metrics: precision {precision}, recall: {recall}, fbeta: {fbeta}")
    #
    # # Compute model metrics on slices
    # # compute_model_metrics_slices(model=model, test_data=test, categorical_features=cat_features, encoder=encoder, lb=lb)

