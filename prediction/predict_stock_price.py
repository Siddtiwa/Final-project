from datetime import timedelta
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
from data.download_data import download_stock_data
from data.data_preprocessing import prepare_lstm_data
from data.get_sentiment_score import get_sentiment_score
from model.model import build_model
import numpy as np
import plotly.graph_objects as go
import plotly
import uuid
import json
import pandas as pd
import tensorflow as tf

def predict_stock_price(ticker, start_date, end_date, model, look_back, units, epochs, batch_size, forecast_days):
    print('start predicting stock price')

    stock_df, error = download_stock_data(ticker, start_date, end_date)
    print(f"Stock data: {stock_df}, Error: {error}")
    if stock_df is None:
        return None, error

    print("debug:1")
    sentiment_df, average_sentiment, sentiment_error = get_sentiment_score(ticker, start_date, end_date)
    if sentiment_df is None:
        return None, f"Sentiment analysis error: {sentiment_error}"
    print("debug:2")
    print(f"sentiment_df:{sentiment_df}")

    x_train, x_test, y_train, y_test, scaler, scaled_data = prepare_lstm_data(pd.DataFrame(sentiment_df), stock_df, look_back)

    print("building model...")
    model = build_model(model, (look_back, 2), units)
    print("model built")
    print("training model...")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    model.fit(train_dataset, epochs=epochs, verbose=0)
    print("model trained")

    # Predictions
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    train_predictions_full = np.zeros((len(train_prediction), 2))
    test_predictions_full = np.zeros((len(test_prediction), 2))
    train_predictions_full[:, 0] = train_prediction.flatten()
    test_predictions_full[:, 0] = test_prediction.flatten()
    train_predictions = scaler.inverse_transform(train_predictions_full)[:, 0]
    test_predictions = scaler.inverse_transform(test_predictions_full)[:, 0]

    # Inverse transform y_train and y_test
    y_train_full = np.zeros((len(y_train), 2))
    y_test_full = np.zeros((len(y_test), 2))
    y_train_full[:, 0] = y_train
    y_test_full[:, 0] = y_test
    y_train = scaler.inverse_transform(y_train_full)[:, 0]
    y_test = scaler.inverse_transform(y_test_full)[:, 0]

    # RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    # Forecasting
    last_sequence = scaled_data[-look_back:].copy()
    forecasts = []
    for _ in range(forecast_days):
        prediction = model.predict(last_sequence.reshape(1, look_back, 2), verbose=0)
        forecasts.append(prediction[0, 0])
        new_row = np.array([prediction[0, 0], last_sequence[-1, -1]])  # Keep sentiment from last
        last_sequence = np.vstack((last_sequence[1:], new_row))

    forecasts_full = np.zeros((len(forecasts), 2))
    forecasts_full[:, 0] = forecasts
    forecasts = scaler.inverse_transform(forecasts_full)[:, 0]

    print("executed till here")
    #forecast_dates = [(stock_df['Date'].iloc[-1] + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
    last_date = pd.to_datetime(stock_df['Date'].iloc[-1])  # Convert just the last date
    forecast_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
    # Visualization
    try:
        fig = go.Figure()
        dates = stock_df['Date'].astype(str).tolist()
        fig.add_trace(go.Scatter(x=dates[look_back:look_back + len(y_train)], y=y_train, name="Actual Train"))
        fig.add_trace(go.Scatter(x=dates[look_back:look_back + len(train_predictions)], y=train_predictions, name="Predicted Train"))
        fig.add_trace(go.Scatter(x=dates[look_back + len(y_train):], y=y_test, name="Actual Test"))
        fig.add_trace(go.Scatter(x=dates[look_back + len(y_train):], y=test_predictions, name="Predicted Test"))
        fig.add_trace(go.Scatter(x=[dates[-1]], y=[stock_df['Close'].iloc[-1]], mode='markers+lines', name="Last Day", line=dict(dash='dash', color='red')))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecasts, mode='lines+markers', name="Forecast", line=dict(color='orange')))
        fig.update_layout(title=f"{ticker} Stock Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        return None, f"Error creating visualization: {str(e)}"

    # Save model
    try:
        model_file = f"{ticker}_stock_price_prediction_{uuid.uuid4().hex}.keras"
        model.save(model_file)
    except Exception as e:
        print("predict_stock_price function does not works")
        return None, f"Error saving model: {str(e)}"

    print("predict_stock_price function does works")

    return {
        'ticker': ticker,
        'avg_sentiment': float(average_sentiment),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'forecast_data': [(date, float(price)) for date, price in zip(forecast_dates, forecasts)],
        'graph_json': graph_json,
        'model_file': model_file
    }, None
