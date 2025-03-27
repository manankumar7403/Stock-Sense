# Stock Price Predictor

This project is an interactive web application built with Streamlit that predicts future stock prices using a hybrid GRU-LSTM neural network model. It fetches historical stock data from Yahoo Finance, enhances it with technical indicators (Close, RSI, MACD, EMA), and provides visualizations and downloadable predictions.

## Features
- Predicts stock prices for a user-specified number of days.
- Uses a GRU-LSTM hybrid model with technical indicators (Close, RSI, MACD, EMA_20).
- Interactive sidebar for configuring prediction settings.
- Visualizes historical vs. predicted prices and training/validation loss.
- Download predictions as a CSV file.

## Requirements
- **Python**: 3.9+ (tested with 3.11)
- **Dependencies**: Listed in `requirements.txt`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mk7403/Stock-Sense.git
   cd Stock-Sense
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages:
   ```
   streamlit==1.44.0
   yfinance==0.2.55
   pandas==2.2.2
   numpy==1.26.4
   scikit-learn==1.5.1
   tensorflow==2.16.1
   matplotlib==3.9.2
   ta==0.11.0
   ```

## Usage
1. **Run the App**:
   ```bash
   streamlit run app.py
   ```

2. **Configure Settings**:
   - In the sidebar, enter a stock ticker (e.g., "AAPL") and adjust the sliders/selectors.
   - Click "Run Prediction" to generate forecasts.

3. **View Results**:
   - See historical vs. predicted price plots, loss curves, a prediction table, and download the results as a CSV.

## Prediction Settings
- **Stock Ticker**: The stock symbol (e.g., "AAPL" for Apple Inc.) to fetch and predict.
- **Historical Data (days)**: Number of past days to fetch (100–1000, default 365). Determines the training dataset size.
- **Days to Predict**: Number of future days to forecast (1–60, default 10).
- **Look-back Period (days)**: Number of past days the GRU-LSTM model uses to predict the next day (10–100, default 60). Defines the sequence length.
- **Training Epochs**: Number of times the model trains on the data (5–50, default 10). Affects learning depth.
- **Batch Size**: Number of samples per training batch (16, 32, 64, 128, default 32). Influences training speed and stability.

## Project Structure
- `app.py`: Main application script.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project Documentation.
- `.gitignore`: Specifies files to exclude from version control.

## Notes
- Data is sourced from Yahoo Finance via `yfinance`.
- The model uses a hybrid GRU (128 units) and LSTM (64 units) architecture with dropout (0.3) for regularization.
- Predictions are estimates; stock markets are inherently unpredictable.
- Random seeds are set (`SEED = 42`) for reproducible results across runs with identical data.

## FAQs

### 1. Why does the fetched data vary slightly even with the same hyperparameters and stock ticker across runs with a short time gap?
**Answer**: The number of fetched rows (e.g., 198 vs. 199) can differ due to the dynamic date range (`datetime.now() - historical_days`). A run might include an extra trading day (e.g., today) depending on the exact timestamp or Yahoo Finance updates. This is normal with live data sources.

### 2. Why do predicted prices vary slightly for the same fetched data, hyperparameters, and stock ticker?
**Answer**: Even though seeds are set (`SEED = 42`) for reproducibility, minor variations may still occur due to factors like floating-point precision differences, GPU computation inconsistencies, or untracked randomness in certain deep-learning operations. If the dataset remains unchanged, predictions should remain largely consistent, but slight fluctuations are normal in neural networks.

### 3. How does changing the number of historical days affect predictions, and what happens when it increases or decreases?
**Answer**: Adjusting `historical_days` changes the dataset size, impacting the model's ability to learn patterns. Increasing it (e.g., from 290 to 500) allows the model to learn from more past data, potentially improving accuracy but also increasing computational complexity and the risk of overfitting. Decreasing it (e.g., from 290 to 100) results in fewer training samples, which can lead to underfitting and unstable predictions, especially for longer forecast periods.
