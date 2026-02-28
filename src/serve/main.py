import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from fastapi import FastAPI
import uvicorn
from datetime import datetime
from pydantic_models import InputPredict, OutputPredict
from mapping import CLASS_NAMES
from bq_io import client, table


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

    insertion_timestamp = datetime.now()
    row = {
        'partition_date': insertion_timestamp.date(),
        'unique_id': input_predict.unique_id,
        'input_features': {
            'sepal_length': input_predict.sepal_length,
            'sepal_width': input_predict.sepal_width,
            'petal_length': input_predict.petal_length,
            'petal_width': input_predict.petal_width,
        },
        'output_results': {
            'class_idx': output.class_idx,
            'class_name': output.class_name,
        },
        'insertion_timestamp': insertion_timestamp,
    }
    client.insert_rows(table, [row])

    return output





if __name__ == '__main__':
    # pipeline = load_pipeline('artifacts/models/model.pkl')
    uvicorn.run(app, host='0.0.0.0', port=8080)