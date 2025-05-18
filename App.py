import uuid
from datetime import datetime
import yfinance as yf
from prediction.predict_stock_price import predict_stock_price
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        data = request.get_json()
        if not request.is_json:
            return jsonify({'error': 'Expected application/json'}), 415
        if not data:
            return jsonify({'error': 'No payload provided.'}), 400

        required_fields = ['ticker', 'start_date', 'end_date']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'missing required fields: {", ".join(missing_fields)}'}), 400

        ticker = data['ticker'].upper()
        start_date = data['start_date']
        end_date = data['end_date']

        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use yyyy-mm-dd'}), 400

        look_back = int(data.get('look_back', 100))
        model_type = data.get('model_type', 'LSTM')
        units = int(data.get('units', 100))
        epochs = int(data.get('epochs', 20))
        batch_size = int(data.get('batch_size', 32))
        forecast_days = int(data.get('forcast_days', 5))

        if model_type not in ['LSTM', 'GRU']:
            return jsonify({'error': 'The model should either be LSTM or GRU'}), 400
        if look_back < 10 or look_back > 100:
            return jsonify({'error': 'Look back should be between 10 and 100 days'}), 400
        if units < 5 or units > 100:
            return jsonify({'error': 'Units should be between 5 and 100'}), 400
        if epochs < 5 or epochs > 50:
            return jsonify({'error': 'Epochs should be between 5 and 50'}), 400
        if batch_size not in [16, 32, 64]:
            return jsonify({'error': 'Batch size should be 16, 32 or 64'}), 400
        if forecast_days < 1 or forecast_days > 30:
            return jsonify({'error': 'Forecast days should be between 1 and 30'}), 400

        # Prediction logic
        result, error = predict_stock_price(
            ticker, start_date, end_date,
            model_type, look_back, units,
            epochs, batch_size, forecast_days
        )

        if result is None:
            return jsonify({'error': f'Failed to process data: {error}'}), 400

        # Fetch extra stock info using yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get("currentPrice")
            previous_close = info.get("previousClose")
            change_percent = None
            if current_price and previous_close:
                change_percent = ((current_price - previous_close) / previous_close) * 100

            stock_data = {
                "sector": info.get("sector"),
                "peRatio": info.get("trailingPE"),
                "marketCap": info.get("marketCap"),
                "dividendYield": (info.get("dividendYield") or 0) * 100,  # convert to %
                "currentPrice": current_price,
                "changePercent": change_percent
            }
        except Exception as e:
            stock_data = {"warning": f"Failed to fetch extra stock info: {str(e)}"}

        return jsonify({
            "result": result,
            "stockInfo": stock_data
        }), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
