import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from fastapi import FastAPI
import uvicorn
from pydantic_models import InputPredict, OutputPredict
from mapping import CLASS_NAMES


def load_pipeline(path: str) -> Pipeline:
    with open(path, 'rb') as f:
        return pickle.load(f)


pipeline = load_pipeline('artifacts/models/model.pkl')


app = FastAPI(
    title='ml-demo-2026',
    version='0.1.0',
)


@app.get('/health-check')
def health_check():
    return 'OK'


@app.post('/predict', response_model=OutputPredict)
def predict(input_predict: InputPredict):
    input_array = np.array([
        input_predict.sepal_length,
        input_predict.sepal_width,
        input_predict.petal_length,
        input_predict.petal_width,
    ]).reshape(1, -1)
    predictions = pipeline.predict(input_array)
    class_idx = predictions.tolist()[0]
    output = OutputPredict(
        class_idx=class_idx,
        class_name=CLASS_NAMES[class_idx]
    )

    return output





if __name__ == '__main__':
    # pipeline = load_pipeline('artifacts/models/model.pkl')
    uvicorn.run(app, host='0.0.0.0', port=8080)