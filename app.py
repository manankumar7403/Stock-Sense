import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ta
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Global features list
FEATURES = ['Close', 'RSI', 'MACD', 'EMA_20']

# Streamlit app configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Advanced Stock Price Prediction App")
st.markdown("""
    Predict future stock prices using a GRU-LSTM hybrid model with technical indicators.
    Enter a stock ticker, configure settings, and visualize results.
""")

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL):", "AAPL").upper()
historical_days = st.sidebar.slider("Historical Data (days):", 100, 1000, 365)
prediction_days = st.sidebar.slider("Days to Predict:", 1, 60, 10)
look_back = st.sidebar.slider("Look-back Period (days):", 10, 100, 60)
epochs = st.sidebar.slider("Training Epochs:", 5, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size:", [16, 32, 64, 128], index=1)

# Cache data fetching to improve performance
@st.cache_data
def fetch_stock_data(ticker, days):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}. Possibly delisted or invalid ticker.")
        st.write(f"Successfully fetched data for {ticker}. Rows: {len(data)}")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Add technical indicators
def add_technical_indicators(df):
    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    return df.dropna()

# Prepare data for GRU-LSTM
def prepare_data(df, look_back, features=FEATURES):
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    if X.size == 0:
        raise ValueError(f"Not enough data: {len(df)} rows < look_back ({look_back})")

    expected_shape = (len(X), look_back, len(features))
    if X.shape != expected_shape:
        raise ValueError(f"X shape {X.shape} doesn’t match expected {expected_shape}")

    return X, y, scaler, scaled_data

# Build and train GRU-LSTM hybrid model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(128, return_sequences=True))  # GRU -> short-term patterns
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))  # LSTM -> long-term patterns
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Predict future prices
def predict_future(model, last_sequence, scaler, days, feature_count):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days):
        current_sequence_reshaped = current_sequence.reshape((1, look_back, feature_count))
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        predictions.append(next_pred[0, 0])

        next_row = current_sequence[-1].copy()
        next_row[0] = next_pred[0, 0]
        current_sequence = np.vstack((current_sequence[1:], next_row))

    pred_array = np.zeros((len(predictions), feature_count))
    pred_array[:, 0] = predictions
    pred_array = scaler.inverse_transform(pred_array)
    return pred_array[:, 0]

# Main app logic
if st.button("Run Prediction"):
    with st.spinner("Processing..."):
        st.write(f"Fetching data for {ticker} with {historical_days} days...")
        raw_data = fetch_stock_data(ticker, historical_days)
        if raw_data is None:
            st.write("Fetch failed. Stopping.")
            st.stop()
        st.write(f"Raw data rows: {len(raw_data)}")

        st.write("Adding technical indicators...")
        data_with_indicators = add_technical_indicators(raw_data)
        if data_with_indicators is None:
            st.write("Indicators failed. Stopping.")
            st.stop()
        st.write(f"Data with indicators rows: {len(data_with_indicators)}")

        st.write("Preparing data for LSTM...")
        X, y, scaler, scaled_data = prepare_data(data_with_indicators, look_back)

        st.write(f"Training samples: {len(X)}")
        if len(X) < batch_size:
            st.warning(f"Batch size ({batch_size}) > samples ({len(X)}). Reducing batch size.")
            batch_size = max(1, len(X) // 2)

        train_size = int(len(X) * 0.8)
        if train_size < 1:
            st.error("Not enough samples for training. Increase historical days.")
            st.stop()

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        st.write("Building and training model...")
        model = build_model((look_back, len(FEATURES)))
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=0)

        st.write("Predicting future prices...")
        last_sequence = scaled_data[-look_back:]
        future_prices = predict_future(model, last_sequence, scaler, prediction_days, len(FEATURES))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data_with_indicators.index, data_with_indicators['Close'], label="Historical Price")
        future_dates = pd.date_range(start=data_with_indicators.index[-1],
                                     periods=prediction_days + 1, freq='B')[1:]
        ax.plot(future_dates, future_prices, label="Predicted Price", color='red', linestyle='--')
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title("Model Loss Over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": future_prices
        })
        st.subheader("Predicted Prices")
        st.dataframe(pred_df)

        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv",
        )

# info
st.sidebar.markdown("""
    ### Notes
    - Uses a GRU-LSTM hybrid model with RSI, MACD, and EMA indicators.
    - Predictions are estimates; stock markets are unpredictable.
    - Data source: Yahoo Finance via `yfinance`.
""")

st.write(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
