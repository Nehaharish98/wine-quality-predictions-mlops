FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install mlflow==2.13.0 # Ensure MLflow is installed here

COPY . .

RUN mkdir -p mlruns \
    && touch mlflow.db

EXPOSE 5000 
EXPOSE 5001 

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api.main:app"]