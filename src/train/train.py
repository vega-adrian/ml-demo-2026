import os
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gcs_io import upload_to_gcs

if os.getenv('ENV', 'local') == 'local':
    from dotenv import load_dotenv
    load_dotenv()


class DemoModel:
    def __init__(
        self,
        random_state: int,
    ):
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        self.is_trained = False

    def _build_pipeline(self):
        pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ]
        )
        return pipeline

    def load_data(self):
        iris = load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=self.random_state,
        )
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        X,
        y,
    ):
        print('Fitting model')
        self.pipeline.fit(X, y)
        print('Model fitted')
        self.is_trained = True

    def evaluate(self, X, y):
        if self.is_trained:
            predictions = self.pipeline.predict(X)
            accuracy = accuracy_score(y, predictions)
            return accuracy
        else:
            raise Exception('The model is not trained')

    def predict(self, X):
        if not self.is_trained:
            raise Exception('The model is not trained')

        return self.pipeline.predict(X)

    def save(self, path: str):
        if not self.is_trained:
            raise Exception('The model is not trained')

        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)

        upload_to_gcs(
            local_file_path=path,
            bucket_name=os.getenv('BUCKET_NAME'),
            destination=os.getenv('DESTINATION_BLOB'),
        )

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.pipeline = pickle.load(f)

        self.is_trained = True


if __name__ == '__main__':
    model = DemoModel(random_state=0)
    X_train, X_test, y_train, y_test = model.load_data()
    model.fit(X_train, y_train)
    model.save('artifacts/models/model.pkl')