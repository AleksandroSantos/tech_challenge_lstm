#!/bin/bash

# Configuração do MLflow Server
export MLFLOW_TRACKING_URI=http://localhost:5000

mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --host 0.0.0.0 \
    --port 5000 \
    --gunicorn-opts "--log-level debug"

