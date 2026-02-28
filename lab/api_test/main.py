import requests


API_HOST = 'https://ml-demo-api-125346888514.europe-west1.run.app'
PREDICT_ENDPOINT = '/predict'


if __name__ == '__main__':
    payload = {
        "unique_id": "string",
        "sepal_length": 0,
        "sepal_width": 0,
        "petal_length": 0,
        "petal_width": 0
    }
    response = requests.post(f"{API_HOST}{PREDICT_ENDPOINT}", json=payload)
    print(response.json())