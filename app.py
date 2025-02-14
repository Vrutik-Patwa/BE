from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
print(tf.__version__)
st.set_page_config(
    page_title="Stock Market Prediction using XAI",
    # page_icon="👋",
)
# st.title('Stock Market Prediction using XAI')
# Predict stock price movement


def assign_risk_category(risk, user_risk_tolerance):
    if risk > 0.035:
        return 'High'
    elif risk > 0.02:
        return 'Moderate'
    else:
        return 'Low'


def predict_momentum(model, scaler, ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='60d')[['Close']]

    if data.isnull().values.any():
        return None  # Handle missing data

    data['Scaled'] = scaler.transform(data[['Close']])
    X = []
    look_back = 30
    for i in range(look_back, len(data)):
        X.append(data['Scaled'].values[i-look_back:i])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    momentum = (predictions[-1] - predictions[0]) / predictions[0]
    return momentum

# user_risk_tolerance = st.selectbox("Choose Your Risk Tolerance", [
#    'Risky', 'Moderate', 'Conservative'])
#
#    import streamlit as st


# Streamlit UI
st.title("Stock Portfolio Generator")

# User selects risk tolerance
user_risk_tolerance = st.selectbox("Select your risk tolerance:", [
                                   "risky", "moderate", "conservative"])

# Available stock tickers
available_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN',
                     'TSLA', 'JNJ', 'NVDA', 'META', 'BA', 'IBM']

# User selects stock tickers
selected_tickers = st.multiselect(
    "Select stocks for portfolio:", available_tickers, default=available_tickers[:5])


def load_trained_model(model_path='lstm_model.h5', scaler_path='scaler.pkl'):
    # model_path = 'tlstm_model.h5'

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def generate_portfolio(user_risk_tolerance, tickers, model, scaler, top_n=5):
    stock_data = []

    for ticker in tickers:
        momentum_score = predict_momentum(model, scaler, ticker)
        if momentum_score is None:
            continue  # Skip stocks with missing data
        risk = np.random.uniform(0.01, 0.05)  # Placeholder risk value
        risk_category = assign_risk_category(risk, user_risk_tolerance)

        stock_data.append({
            'Ticker': ticker,
            'Momentum': momentum_score,
            'Risk': risk,
            'RiskCategory': risk_category
        })

    stock_df = pd.DataFrame(stock_data)
    if user_risk_tolerance == 'risky':
       filtered_stocks = stock_df[stock_df['RiskCategory'] == 'High']
    elif user_risk_tolerance == 'moderate':
      filtered_stocks = stock_df[stock_df['RiskCategory'] == 'Moderate']
    elif user_risk_tolerance == 'conservative':
      filtered_stocks = stock_df[stock_df['RiskCategory'].isin(['Low', 'Moderate'])]
    else:
      filtered_stocks = stock_df

    filtered_stocks = filtered_stocks.sort_values(
        by='Momentum', ascending=False)

    return filtered_stocks['Ticker'].head(top_n).tolist()


if st.button("Generate Portfolio"):
    # Load trained model and scaler
    model, scaler = load_trained_model()

    # Generate portfolio based on user input
    portfolio = generate_portfolio(
        user_risk_tolerance, selected_tickers, model, scaler, top_n=5)

    df = pd.DataFrame(portfolio, columns=["Stock Ticker"])
    st.write(f"Optimal portfolio for {user_risk_tolerance} risk tolerance:")
    st.table(df)
