import logging as log
import sys
from starter.starter.ml.model import inference, load_model, compute_model_metrics_slices, compute_model_metrics
from starter.starter.ml.data import process_data, load_data
from starter.starter.train_model import cat_features


if __name__ == "__main__":
    log.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    model = load_model("logreg_model.pkl")
    encoder = load_model("encoder.pkl")
    label_binarizer = load_model("label_binarizer.pkl")
    scaler = load_model("scaler.pkl")

    test_data = load_data("test_data.csv")

    # Process the test data with the process_data function.
    X_test, y_test, _, _, _ = process_data(test_data, categorical_features=cat_features, label='salary',
                                           training=False,
                                           encoder=encoder, lb=label_binarizer, scaler=scaler)

    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
    log.info(f"Model metrics on test data: precision {precision}, recall: {recall}, fbeta: {fbeta}")

    compute_model_metrics_slices(model=model, test_data=test_data, categorical_features=cat_features, encoder=encoder,
                                 lb=label_binarizer, scaler=scaler)
