import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.keras
import logging
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense

import uuid


# Configuração do logger
logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(uuid.uuid4().hex)  # create a new experiment and use it
with mlflow.start_run():
    print(mlflow.get_artifact_uri())


def download_data(stock_symbol: str, start_date: str, end_date: str) -> np.ndarray:
    """
    Baixa os dados históricos de fechamento de uma ação usando yfinance.

    Parâmetros:
        stock_symbol (str): Símbolo da ação.
        start_date (str): Data inicial no formato 'YYYY-MM-DD'.
        end_date (str): Data final no formato 'YYYY-MM-DD'.

    Retorna:
        np.ndarray: Array com os valores de fechamento.
    """
    logging.info(f"Baixando dados para {stock_symbol}...")
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("Nenhum dado retornado.")
        return data["Close"].values.reshape(-1, 1)
    except Exception as e:
        logging.error(f"Erro ao baixar dados: {e}")
        return None


def preprocess_data(data):
    if data is None:
        raise ValueError("Os dados não podem ser None.")

    if np.isnan(data).any():
        logging.warning(
            "Dados contêm valores ausentes. Realizando preenchimento com ffill."
        )
        data = (
            pd.DataFrame(data).fillna(method="ffill").values
        )  # Preenche com último valor válido

    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)


def create_time_series_data(data: np.ndarray, n_steps: int):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i : (i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(x), np.array(y)


def build_lstm_model(units, input_shape, optimizer):
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=input_shape),
            LSTM(units),
            Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler: MinMaxScaler):
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}


def log_metrics(metrics):
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)


def train_and_evaluate(x_train, y_train, x_val, y_val, scaler, param_grid):
    logging.info("Iniciando treinamento e avaliação...")

    best_mape = float("inf")
    best_params = {}
    best_model = None

    for params in param_grid:
        with mlflow.start_run():
            mlflow.log_params(params)

            model = build_lstm_model(
                params["units"], (x_train.shape[1], 1), params["optimizer"]
            )
            history = model.fit(
                x_train,
                y_train,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                verbose=0,
                validation_data=(x_val, y_val),
            )

            # Loga as métricas de perda
            for epoch in range(len(history.history["loss"])):
                mlflow.log_metric(
                    "train_loss", history.history["loss"][epoch], step=epoch
                )
                mlflow.log_metric(
                    "val_loss", history.history["val_loss"][epoch], step=epoch
                )

            val_predictions = model.predict(x_val)
            metrics = calculate_metrics(y_val, val_predictions, scaler)

            # Logar métricas
            log_metrics(metrics)
            # mlflow.keras.log_model(model, "lstm_model")

            if metrics["MAPE"] < best_mape:
                best_mape = metrics["MAPE"]
                best_params = params
                best_model = model

    logging.info(
        f"Treinamento concluído. Melhores parâmetros: {best_params}, Melhor MAPE: {best_mape:.2f}%"
    )
    return best_model, best_params


def main():

    stock_symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-11-30"

    closing_prices = download_data(stock_symbol, start_date, end_date)
    if closing_prices is None:
        logging.error(
            "Nenhum dado foi baixado. Verifique o símbolo da ação ou a conexão com a internet."
        )
        return

    scaler, closing_prices_scaled = preprocess_data(closing_prices)

    n_steps = 60
    x_data, y_data = create_time_series_data(closing_prices_scaled, n_steps)

    split = int(0.8 * len(x_data))
    x_train, x_val = x_data[:split], x_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    param_grid = [
        {"epochs": 50, "batch_size": 16, "units": 50, "optimizer": "adam"},
        {"epochs": 100, "batch_size": 32, "units": 100, "optimizer": "rmsprop"},
    ]

    best_model, _ = train_and_evaluate(
        x_train, y_train, x_val, y_val, scaler, param_grid
    )
    best_model.save("lstm_stock_model_best.h5")


if __name__ == "__main__":
    main()
