import pickle
import numpy as np
from sklearn.pipeline import Pipeline


def load_pipeline(path: str) -> Pipeline:
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    pipeline = load_pipeline('artifacts/models/model.pkl')

    X = [5.1, 3.4, 1.5, 0.2]
    pred = pipeline.predict(np.array(X).reshape(1, -1))
    print(pred)