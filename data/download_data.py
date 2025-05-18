import yfinance as yf
from datetime import datetime
import pandas as pd

def parse_date(date_str):
    """Try parsing date in both yyyy-mm-dd and yy-mm-dd formats."""
    for fmt in ('%Y-%m-%d', '%y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    raise ValueError('Invalid date format. Use yyyy-mm-dd or yy-mm-dd')

def download_stock_data(ticker, start_date, end_date):
    try:
        if not ticker or not isinstance(ticker, str):
            return None, 'Invalid ticker symbol'
        try:
            # Validate and convert dates
            start = parse_date(start_date)
            end = parse_date(end_date)
        except ValueError as e:
            return None, str(e)

        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval='1d')

        if df.empty:
            return None, f'No data found for {ticker} between {start_date} and {end_date}'

        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df[['Date', 'Close']]

        print("download_stock_data function works")
        print(type(df))
        return df, None
    except Exception as e:
        print("download_stock_data function does not work")
        return None, f"Error downloading data for {ticker}: {str(e)}"
