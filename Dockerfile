# Dockerfile para o projeto LSTM com FastAPI
FROM python:3.9-slim AS api-service

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY model/lstm_stock_model_best.h5 /model/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
