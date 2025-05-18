# Final-project
# 📈 Stock Market Prediction and Analysis Platform

A full-stack web application that provides real-time stock insights, historical data analysis, financial metrics, personalized portfolio tracking, and future price predictions using deep learning models like LSTM and GRU. This platform integrates a React frontend, Flask backend, and leverages `yfinance` for live stock data and custom ML models for predictive analytics.

---

## 🚀 Features

### 🔍 Stock Insights
- **Current price**, **% change**, **P/E ratio**, **sector**, **market cap**, **dividend yield**, and more — fetched live via `yfinance`.

### 📊 Historical Trends
- View past stock performance with interactive charts.
- Filter by stock ticker.

### 📉 Financial Metrics
- Compare key financial metrics across companies and sectors.
- Assess valuation and fundamentals.

### 💼 Portfolio Tracker
- Track owned shares, average buy price, current value, and overall gain/loss.

### 🤖 Price Prediction
- Predict stock prices using deep learning models (LSTM or GRU).
- Customizable parameters: look-back period, epochs, batch size, etc.
- Forecast up to 30 days into the future.

---

## 🧠 Tech Stack

| Layer         | Tech Used                                 |
|--------------|--------------------------------------------|
| **Frontend**  | React, Redux Toolkit, TypeScript, Chart.js |
| **Backend**   | Flask, Python, yfinance, NumPy, Pandas     |
| **ML Models** | LSTM / GRU via Keras/TensorFlow            |
| **Database**  | In-memory (optional integration planned)   |

---

## ⚙️ API Endpoints

### `GET /stocks/historical`
Fetch historical data for a stock.

### `GET /stocks/metrics`
Retrieve financial metrics (P/E ratio, dividend yield, etc.).

### `GET /stocks/portfolio`
Returns mock or user-added portfolio data.

### `GET /stocks/summary`
Overall market summary.

### `GET /stocks/predictions`
Returns previous model predictions (static or cached).

### `POST /predict`
Trains a deep learning model and returns future predicted prices.  
**Body:**
```json
{
  "ticker": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2023-01-01",
  "look_back": 60,
  "model_type": "LSTM",
  "units": 100,
  "epochs": 20,
  "batch_size": 32,
  "forcast_days": 5
}
