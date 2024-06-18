import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from indicators import calculate_indicators
import logging

# Defina a chave de API do CoinMarketCap e NewsAPI
CMC_API_KEY = '21366b12-acc0-4843-8769-9b48e100c9b7'
NEWS_API_KEY = '89f83b31dae3493289dcf46f9e60b0e5'



def get_crypto_data():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': CMC_API_KEY,
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return data['data']


def load_historical_data(symbol):
    url = f'https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical?symbol={symbol}&convert=USD&time_start=1622505600&time_end=1625097600'  # Ajuste o intervalo de tempo conforme necessário
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': CMC_API_KEY,
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    logging.debug(f"Response from CoinMarketCap API: {data}")

    if 'data' in data and 'quotes' in data['data']:
        quotes = data['data']['quotes']
        prices = [quote['quote']['USD']['close'] for quote in quotes]
        df = pd.DataFrame(prices, columns=['close'])
        df['date'] = pd.to_datetime([quote['timestamp'] for quote in quotes])
        df.set_index('date', inplace=True)

        # Calcular indicadores técnicos
        df = calculate_indicators(df)

        return df
    else:
        logging.error(f"Error fetching data from API: {data}")
        raise KeyError('data.quotes')


def get_news_sentiment():
    url = f'https://newsapi.org/v2/everything?q=bitcoin&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data['articles']

    # Exemplo simples de análise de sentimentos usando TextBlob
    from textblob import TextBlob
    sentiments = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        if title and description:
            analysis = TextBlob(title + " " + description)
            sentiment = analysis.sentiment.polarity
            sentiments.append(sentiment)

    # Retornar a média dos sentimentos
    if sentiments:
        return sum(sentiments) / len(sentiments)
    else:
        return 0  # Caso não haja sentimentos para analisar


def get_news_articles():
    url = f'https://newsapi.org/v2/everything?q=bitcoin&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()
    articles = data['articles'][:3]  # Pegar as três primeiras notícias
    return [{'title': article['title'], 'description': article['description'], 'url': article['url']} for article in
            articles]


def train_model():
    df = load_historical_data()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close', 'SMA', 'EMA', 'RSI', 'MACD']].values)

    prediction_days = 60
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x])
        y_train.append(scaled_data[x, 0])  # Preço de fechamento como variável alvo

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Salvar o modelo e o scaler
    model.save('crypto_model.keras')
    joblib.dump(scaler, 'crypto_scaler.pkl')

    return model, scaler


def make_prediction(model, scaler, prices):
    inputs = np.array(prices).reshape(-1, 5)  # 5 características: close, SMA, EMA, RSI, MACD
    inputs = scaler.transform(inputs)

    X_test = []
    X_test.append(inputs[-60:])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]


def load_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 5)))  # 5 características
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.load_weights('crypto_model.keras')
    scaler = joblib.load('crypto_scaler.pkl')
    return model, scaler
