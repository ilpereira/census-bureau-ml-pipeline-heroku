import logging as log
import sys
from starter.starter.ml.model import load_model, compute_model_metrics_slices
from starter.starter.ml.data import process_data, load_data
from starter.starter.train_model import cat_features

model = load_model("logreg_model.pkl")
encoder = load_model("encoder.pkl")
label_binarizer = load_model("label_binarizer.pkl")
scaler = load_model("scaler.pkl")

if __name__ == "__main__":
    log.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    test_data = load_data("test_data.csv")

    # Process the test data with the process_data function.
    X_test, y_test, _, _, _ = process_data(test_data, categorical_features=cat_features, label='salary',
                                           training=False,
                                           encoder=encoder, lb=label_binarizer, scaler=scaler)

    compute_model_metrics_slices(model=model, test_data=test_data, categorical_features=cat_features, encoder=encoder,
                                 lb=label_binarizer, scaler=scaler)
