import pandas as pd
from fastapi import FastAPI
from starter.starter.ml.model import inference, load_model
from starter.starter.ml.data import process_data
from pydantic import BaseModel, Field
import uvicorn
from typing import Union, List, Optional
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID --profile personal-aws-profile")
    os.system("aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY --profile personal-aws-profile")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

model = load_model("logreg_model.pkl")
encoder = load_model("encoder.pkl")
label_binarizer = load_model("label_binarizer.pkl")
scaler = load_model("scaler.pkl")

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


class CensusModel(BaseModel):
    age: Optional[Union[int, List[int]]]
    workclass: Optional[Union[str, List[str]]]
    fnlgt: Optional[Union[int, List[int]]]
    education: Optional[Union[str, List[str]]]
    education_num: Optional[Union[int, List[int]]] = Field(alias="education-num")
    marital_status: Optional[Union[str, List[str]]] = Field(alias="marital-status")
    occupation: Optional[Union[str, List[str]]]
    relationship: Optional[Union[str, List[str]]]
    race: Optional[Union[str, List[str]]]
    sex: Optional[Union[str, List[str]]]
    capital_gain: Optional[Union[int, List[int]]] = Field(alias="capital-gain")
    capital_loss: Optional[Union[int, List[int]]] = Field(alias="capital-loss")
    hours_per_week: Optional[Union[float, List[float]]] = Field(alias="hours-per-week")
    native_country: Optional[Union[str, List[str]]] = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": [31, 37],
                "workclass": ["Private", "Federal-gov"],
                "fnlgt": [125421, 16235],
                "education": ["Bachelors", "HS-grad"],
                "education_num": [8, 14],
                "marital_status": ["Divorced", "Married-civ-spouse"],
                "occupation": ["", ""],
                "relationship": ["Unmarried", "Wife"],
                "race": ["Asian-Pac-Islander", "Black"],
                "sex": ["Male", "Female"],
                "capital_gain": [0, 5013],
                "capital_loss": [0, 0],
                "hours_per_week": [40, 35],
                "native_country": ["India", "United-States"]
            }
        }
        allow_population_by_field_name = True


# Instantiate the app.
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def welcome():
    return {"greeting": "Welcome!"}


@app.post("/inference/")
async def model_inference(item: CensusModel):
    data_dict = item.dict(by_alias=True)
    for key, value in data_dict.items():
        if not isinstance(data_dict[key], list):
            data_dict[key] = [value]

    df = pd.DataFrame(data_dict)

    data, _, _, _, _ = process_data(X=df, categorical_features=cat_features, label=None, training=False,
                                    encoder=encoder, lb=label_binarizer, scaler=scaler)

    inference_result = inference(model=model, X=data)
    inference_result_converted = label_binarizer.inverse_transform(inference_result)

    return {"prediction": list(inference_result_converted)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
