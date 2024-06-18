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
