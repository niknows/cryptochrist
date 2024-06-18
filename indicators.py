import pandas as pd
import pandas_ta as ta

def calculate_indicators(df):
    # Calcular indicadores técnicos
    df['SMA'] = ta.sma(df['close'], length=14)
    df['EMA'] = ta.ema(df['close'], length=14)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'] = ta.macd(df['close'], fast=12, slow=26)['MACD_12_26_9']
    df.dropna(inplace=True)
    return df
