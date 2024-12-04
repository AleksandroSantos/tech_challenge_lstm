# Dockerfile para o servidor MLflow
FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow

RUN chmod -R 777 /app

COPY mlflow_server.sh .
RUN chmod +x mlflow_server.sh

CMD ["./mlflow_server.sh"]
