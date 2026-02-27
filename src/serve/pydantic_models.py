from pydantic import BaseModel


class InputPredict(BaseModel):
    unique_id: str
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class OutputPredict(BaseModel):
    class_idx: int
    class_name: str