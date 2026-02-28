FROM python:3.13-slim

WORKDIR /app

RUN pip install poetry

COPY poetry.lock /app
COPY pyproject.toml /app

RUN poetry install --without local,test,serve

RUN mkdir /app/src
RUN mkdir /app/artifacts
RUN mkdir /app/artifacts/models

COPY src/train /app/src/train

CMD ["poetry", "run", "python", "src/train/train.py"]