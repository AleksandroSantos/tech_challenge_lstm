# Projeto de Previsão de Preços de Ações com LSTM e FastAPI

Este projeto utiliza redes neurais LSTM (Long Short-Term Memory) para prever o valor de fechamento de ações com base em dados históricos. A solução é dividida em duas partes principais:
1. **Treinamento do Modelo LSTM**: Utiliza dados históricos de ações para treinar o modelo.
2. **API FastAPI**: Permite que os usuários façam previsões de preços de ações via uma API RESTful.

A solução é containerizada usando Docker, com o servidor **MLflow** para rastreamento de experimentos e gerenciamento de modelos.

---

## Funcionalidades do Projeto LSTM para Previsão de Ações

### 1. **Coleta e Pré-processamento de Dados**
   - Utiliza a biblioteca **`yfinance`** para obter dados históricos de ações.
   - Realiza limpeza e transformação dos dados, incluindo normalização.

### 2. **Treinamento do Modelo LSTM**
   - Implementa uma rede neural **LSTM** para previsão de valores de fechamento das ações.
   - Inclui ajuste de hiperparâmetros para otimizar o desempenho.

### 3. **Validação e Avaliação do Modelo**
   - Avalia o modelo com métricas como **RMSE** e **MAE**.
   - Utiliza uma parte dos dados para **validação cruzada**.

### 4. **Registro e Rastreamento de Experimentos**
   - Integração com **MLflow** para rastreamento de experimentos.
   - Registra modelos treinados e suas métricas de desempenho.

### 5. **Deploy da API com FastAPI**
   - Desenvolve uma **API RESTful** usando **FastAPI** para realizar previsões em tempo real.
   - Permite entrada de dados históricos e retorna previsões futuras.

### 6. **Containerização com Docker**
   - Utiliza **Docker** para facilitar o deploy e escalabilidade da aplicação.
   - Inclui scripts **Dockerfile** para automação da construção da imagem.

### 7. **Monitoramento em Produção**
   - Configuração para monitorar a **performance** do modelo em ambiente de produção.
   - Possibilita atualizar e versionar o modelo de forma eficiente.

---

## Estrutura do Projeto

```plaintext
tech_challenge_lstm/
│
├── app/                   # API FastAPI
│   ├── main.py            # Script principal da API
│   └── requirements.txt   # Dependências para API FastAPI
│
├── mlflow_setup/
│   ├── Dockerfile         # Dockerfile para o servidor MLflow
│   └── mlflow_server.sh   # Script para iniciar o servidor MLflow
│
├── model/                 # Treinamento do modelo LSTM
│   ├── train_model.py     # Script para treinar o modelo LSTM
│   └── requirements.txt   # Dependências para treinamento
│
├── docker-compose.yml     # Orquestra os contêineres (MLflow e API)
├── Dockerfile             # Dockerfile principal para o projeto
└── README.md              # Documentação do projeto
```

## Configuração e Execução

### **1. Pré-requisitos**

- Python 3.9 ou superior  
- Docker (para deploy)  
- Pipenv ou virtualenv (opcional, mas recomendado)  

### **2. Instalação das Dependências**

#### **Instalação Local**

1. Clone este repositório:
   ```bash
   git clone https://github.com/AleksandroSantos/tech_challenge_lstm.git
   cd tech_challenge_lstm
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python3 -m venv .venv
   source venv/bin/activate  # No Windows use `venv\Scripts\activate`
   ```

3. Construir os containers Docker:
   ```bash
      docker-compose build
   ```

4. Inicie os containers:
   ```bash
      docker-compose up
   ```

5. MLflow estará disponível em:
   ```bash
    http://localhost:5000
   ```

### **3. Treinamento do Modelo**

1. Edite o arquivo train_model.py se quiser mudar o símbolo da ação ou datas.
2. Instale as dependências:
```bash
   cd model/
   pip install -r requirements.txt
```
3. Execute o script de treinamento:
   ```bash
    python model/train_model.py
   ```

### **4. Acessar a API:**

1. A API FastAPI estará disponível em:
   ```bash
    http://localhost:8000
   ```
   
2. Documentação interativa da API (Swagger UI) disponível em:
   ```bash
   http://localhost:8000/docs
   ```

### **5. Testando a API**

Você pode usar o Postman ou curl para testar a API:

   ```bash
      curl -X 'POST' \
      'http://localhost:8000/predict/' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "stock_symbol": "AAPL",
      "start_date": "2024-10-01",
      "end_date": "2024-11-30",
      "days_ahead": 10
      }'
   ```
