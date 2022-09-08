import requests
import json

data_1 = {"age": 31, "workclass": "Private", "fnlgt": 117000, "education": "Doctorate", "education_num": 15,
          "marital_status": "Divorced", "occupation": "Prof-specialty", "relationship": "Husband", "race": "White",
          "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States"}

data_2 = {"age": 52, "workclass": "Self-emp-not-inc", "fnlgt": 209642, "education": "HS-grad", "education_num": 9,
          "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial", "relationship": "Husband",
          "race": "White", "sex": "Male", "capital_gain": 0, "capital_loss": 0, "hours_per_week": 45,
          "native_country": "United-States"}

data_list = {"age": [31, 52], "workclass": ["Private", "Self-emp-not-inc"], "fnlgt": [117000, 209642],
             "education": ["Doctorate", "HS-grad"], "education_num": [15, 9],
             "marital_status": ["Divorced", "Married-civ-spouse"], "occupation": ["Prof-specialty", "Exec-managerial"],
             "relationship": ["Husband", "Husband"], "race": ["White", "White"], "sex": ["Male", "Male"],
             "capital_gain": [0, 0], "capital_loss": [0, 0], "hours_per_week": [40, 45],
             "native_country": ["United-States", "United-States"]}

# url = "http://127.0.0.1:8000/inference/"
url = "https://census-bureau-ml-pipeline.herokuapp.com/inference/"
req = requests.post(url=url, data=json.dumps(data_list))

print(req.json())
