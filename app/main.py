from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = FastAPI()

# Carregar o modelo treinado
model = load_model("../model/lstm_stock_model_best.h5")

scaler = MinMaxScaler(feature_range=(0, 1))


# Modelo de dados de entrada
class StockData(BaseModel):
    stock_symbol: str
    start_date: str
    end_date: str
    days_ahead: int


# Função para fazer previsões
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

    days_ahead = stock_data.days_ahead

    predictions = {}
    for i in range(days_ahead):

        # Transforma os dados de input
        to_predict = np.array(closing_prices[-3:]).reshape(-1, 1)
        to_predict = scaler.fit_transform(to_predict)
        to_predict = to_predict.reshape(1, 3, 1)

        # Realiza a predição
        prediction = model.predict(to_predict)

        # Realiza a inversão do valor
        prediction = scaler.inverse_transform(prediction)
        predictions[f"prediction_day_{i+1}"] = prediction[0, 0]
        
    # Retorna a previsão
    return list(predictions)
