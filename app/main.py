import numpy as np
import yfinance as yf
from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Carregar o modelo treinado
model = load_model("../model/lstm_stock_model_best.h5")

# Instanciar o scaler
scaler = MinMaxScaler(feature_range=(0, 1))


# Modelo de dados de entrada
class StockData(BaseModel):
    stock_symbol: str
    start_date: str
    end_date: str
    days_ahead: int


# Função para preparar os dados
def prepare_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    closing_prices = data["Close"].values.reshape(-1, 1)
    return closing_prices


# Rota de previsão
@app.post("/predict/")
async def predict(stock_data: StockData):
    closing_prices = prepare_data(
        stock_data.stock_symbol,
        stock_data.start_date,
        stock_data.end_date,
    )

    # Normalizar os dados
    scaled_data = scaler.fit_transform(closing_prices)

    # Últimos dados para iniciar as previsões
    input_sequence = scaled_data[-3:].reshape(
        1, 3, 1
    )  # Assume que a LSTM usa uma sequência de 3 timesteps

    predictions = []

    for i in range(stock_data.days_ahead):
        # Realiza a predição
        prediction = model.predict(input_sequence)

        # Inversão para o valor original e conversão para float
        prediction_original = float(scaler.inverse_transform(prediction)[0, 0])
        predictions.append(prediction_original)

        # Adiciona a previsão atual na sequência para a próxima previsão
        new_input = np.append(
            input_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1
        )
        input_sequence = new_input

    # Retorna a lista de previsões como JSON serializável
    return {"predictions": predictions}
