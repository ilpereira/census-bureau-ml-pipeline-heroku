import requests
import json

data = {"age": 31, "workclass": "Private", "fnlgt": 117000, "education": "Doctorate", "education_num": 15,
        "marital_status": "Divorced", "occupation": "Prof-specialty", "relationship": "Husband", "race": "White",
        "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States"}

req = requests.post(url="http://127.0.0.1:8000/inference/", data=json.dumps(data))

print(req.json())
