def load_historical_data(symbol='bitcoin'):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency=usd&days=365&interval=daily'
    response = requests.get(url)
    data = response.json()

    if 'prices' in data:
        prices = [item[1] for item in data['prices']]
        df = pd.DataFrame(prices, columns=['close'])
        df['date'] = pd.date_range(start=pd.Timestamp.today() - pd.DateOffset(days=len(df) - 1), periods=len(df))
        df.set_index('date', inplace=True)

        # Calcular indicadores técnicos
        df = calculate_indicators(df)

        return df
    else:
        raise KeyError('prices')
