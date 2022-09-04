import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome!"}


def test_inference_over_50k():
    data = {"age": 52, "workclass": "Self-emp-not-inc", "fnlgt": 209642, "education": "HS-grad", "education_num": 9,
            "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial", "relationship": "Husband",
            "race": "White", "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 45,
            "native_country": "United-States"}

    r = client.post("/inference/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"prediction": [">50K"]}


def test_inference_less_than_equal_50k():
    data = {"age": 49, "workclass": "Private", "fnlgt": 160187, "education": "9th", "education_num": 5,
            "marital_status": "Married-spouse-absent", "occupation": "Other-service", "relationship": "Not-in-family",
            "race": "Black", "sex": "Female", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 16,
            "native_country": "Jamaica"}

    r = client.post("/inference/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"prediction": ["<=50K"]}
