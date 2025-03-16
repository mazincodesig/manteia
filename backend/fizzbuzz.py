from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON from the request body
        data = request.get_json(silent=True)
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Invalid request format. Ensure JSON is sent with Content-Type: application/json'}), 400

        ticker = data.get('ticker', '').strip().upper()
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400

        # Fetch stock data for the past 30 days
        today = datetime.today().date()
        start_date = today - timedelta(days=30)
        stock_data = yf.download(ticker, start=start_date, end=today)

        if stock_data.empty:
            return jsonify({'error': f'No stock data available for {ticker}'}), 400

        # Feature Engineering
        stock_data['Close_Lag1'] = stock_data['Close'].shift(1)
        stock_data['Close_Lag2'] = stock_data['Close'].shift(2)
        stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['Volatility'] = stock_data['High'] - stock_data['Low']
        stock_data.dropna(inplace=True)

        if stock_data.empty:
            return jsonify({'error': 'Not enough data after preprocessing.'}), 400

        # Defining Features and Target (X = features, y = target)
        X = stock_data[['Close_Lag1', 'Close_Lag2', 'SMA_5', 'Volatility']]
        y = stock_data['Close'].values

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Train SVR with Hyperparameter Tuning
        param_grid = {
            'C': [1, 10, 100, 1000],
            'gamma': [0.01, 0.1, 'scale'],
            'epsilon': [0.001, 0.01, 0.1, 0.5]
        }

        grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_svr_params = grid_search.best_params_

        svr_model = SVR(kernel='rbf', **best_svr_params)
        svr_model.fit(X_train, y_train)

        # Train Gradient Boosting Regressor
        gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
        gbr_model.fit(X_train, y_train)

        # Predict using the most recent data
        latest_data = scaler_X.transform(np.array([
            stock_data['Close_Lag1'].iloc[-1],
            stock_data['Close_Lag2'].iloc[-1],
            stock_data['SMA_5'].iloc[-1],
            stock_data['Volatility'].iloc[-1]
        ]).reshape(1, -1))

        svr_prediction = svr_model.predict(latest_data)[0]
        gbr_prediction = gbr_model.predict(latest_data)[0]

        # Convert predictions back to original scale
        svr_prediction = scaler_y.inverse_transform([[svr_prediction]])[0, 0]
        gbr_prediction = scaler_y.inverse_transform([[gbr_prediction]])[0, 0]

        return jsonify({
            'svr_prediction': round(svr_prediction, 2),
            'gbr_prediction': round(gbr_prediction, 2),
            'svr_mse': round(mean_squared_error(y_test_scaled, svr_model.predict(X_test)), 4),
            'gbr_mse': round(mean_squared_error(y_test_scaled, gbr_model.predict(X_test)), 4),
            'best_svr_params': best_svr_params
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/plot', methods=['POST'])
def get_stock_history():
    try:
        data = request.get_json()
        ticker = data.get("ticker", "").strip().upper()
        if not ticker:
            print("❌ Error: No ticker provided")
            return jsonify({"error": "Ticker symbol is required"}), 400

        today = datetime.today().date()
        start_date = today - timedelta(days=30)

        print(f"📊 Fetching stock data for: {ticker} (from {start_date} to {today})")

        stock_data = yf.download(ticker, start=start_date, end=today)

        if stock_data.empty:
            print(f"❌ Error: No stock data available for {ticker}")
            return jsonify({"error": f"No stock data available for {ticker}"}), 400

        # Flatten columns in case they are multi-index (tuple keys)
        stock_data.columns = [col if isinstance(col, str) else col[0] for col in stock_data.columns]

        # Prepare historical data: reset index to bring 'Date' into a column,
        # then format the date and rename "Close" to "price" for consistency.
        stock_data.reset_index(inplace=True)
        stock_data["Date"] = stock_data["Date"].dt.strftime('%Y-%m-%d')
        historical_prices = stock_data[["Date", "Close"]].rename(columns={"Close": "price"})

        prices_list = historical_prices.to_dict(orient="records")
        print("✅ Successfully fetched historical data:", prices_list[:5])  # Log first 5 records

        return jsonify({"prices": prices_list})
    except Exception as e:
        print("❌ INTERNAL SERVER ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
