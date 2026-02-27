FROM python:3.13-slim

WORKDIR /app

RUN pip install poetry

COPY poetry.lock /app
COPY pyproject.toml /app

RUN poetry install

RUN mkdir /app/src
RUN mkdir /app/artifacts
RUN mkdir /app/artifacts/models

COPY artifacts/models/model.pkl /app/artifacts/models/
COPY src/serve /app/src/serve

CMD ["python", "src/serve/main.py"]