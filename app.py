from flask import Flask, render_template, request, jsonify
from model import make_prediction, get_crypto_data, get_news_sentiment, load_model, load_historical_data, get_news_articles
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Carregar o modelo salvo na inicialização do servidor
model, scaler = load_model()

@app.route('/')
def index():
    crypto_data = get_crypto_data()
    sentiment = get_news_sentiment()
    news = get_news_articles()
    return render_template('index.html', crypto_data=crypto_data, sentiment=sentiment, news=news)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    logging.debug(f"Received data for prediction: {data}")
    prediction = make_prediction(model, scaler, data['prices'])
    logging.debug(f"Prediction result: {prediction}")
    return jsonify({'predicted_price': prediction})

@app.route('/fetch-prices/<symbol>', methods=['GET'])
def fetch_prices(symbol):
    try:
        logging.debug(f"Fetching prices for symbol: {symbol}")
        historical_data = load_historical_data(symbol)
        logging.debug(f"Historical data for {symbol}: {historical_data}")
        prices = historical_data['close'].tail(60).tolist()
        logging.debug(f"Fetched prices: {prices}")
        return jsonify({'prices': prices})
    except KeyError as e:
        logging.error(f"Error fetching prices: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
