import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(sentiment_df, stock_df , look_back):
    #print("preparing data...")
    #print(type(sentiment_df))
    #print(sentiment_df)
    print(type(stock_df))
    print(stock_df)
    #print(sentiment_df)
    #print(stock_df)
    try:
        stock_df = pd.DataFrame(stock_df)
        sentiment_df = pd.DataFrame(sentiment_df)
        # Validate inputs
        if not isinstance(stock_df, pd.DataFrame) or not isinstance(sentiment_df, pd.DataFrame):
            print("Debugging line a")
            return None, None, None, None, None, None, "Invalid input: stock_df and sentiment_df must be pandas DataFrames"
        if not all(col in stock_df.columns for col in ['Date', 'Close']):
            print("Debugging line b")
            return None, None, None, None, None, None, "stock_df must contain 'Date' and 'Close' columns"
        if not all(col in sentiment_df.columns for col in ['Date', 'Sentiment']):
            print("Debugging line c")
            return None, None, None, None, None, None, "sentiment_df must contain 'Date' and 'Sentiment' columns"
        if not isinstance(look_back, int) or look_back < 10 or look_back > 200:
            print("Debugging line d")
            return None, None, None, None, None, None, "look_back must be an integer between 10 and 200"
        if len(stock_df) < look_back:
            print("Debugging line e")
            return None, None, None, None, None, None, f"stock_df has fewer rows ({len(stock_df)}) than look_back ({look_back})"

        # Merge stock and sentiment data on Date
        print("Debugging line f")
        data = stock_df.merge(sentiment_df, on='Date', how='left')
        data['Sentiment'] = data['Sentiment'].fillna(0)  # Fill missing sentiment with 0

        # Prepare features (Close price and Sentiment)
        print("Debugging line g")
        features = data[['Close', 'Sentiment']].values

        # Scale features
        print("Debugging line h")
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features)

        # Create sequences
        print("Debugging line i")
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        X, y = np.array(X), np.array(y)

        # Split into train and test (80-20 split)
        print("Debugging line j")
        train_size = int(len(X) * 0.8)
        if train_size == 0:
            print("Debugging line k")
            return None, None, None, None, None, None, "Insufficient data for training after sequence creation"
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print("prepare_LSTM_data function works")
        return X_train, X_test, y_train, y_test, scaler, scaled_data
    except Exception as e:
        print("prepare_lstm_data function does not works")
        print(str(e))
        return None, None, None, None, None, None, f"Error preparing data: {str(e)}"
