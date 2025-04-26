#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MUST be first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Eventfolio - Economic Event Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard library imports
import base64
from datetime import datetime, timedelta
from io import BytesIO
import warnings

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

# --------------------- DATA LOADING ---------------------
@st.cache_data
def load_data():
    try:
        economic_events = pd.read_excel('Economic Event.xlsx')
        industry_stocks = pd.read_excel('Industry_EconomicEvent_Stocks.xlsx')
        
        economic_events.columns = economic_events.columns.str.strip()
        industry_stocks.columns = industry_stocks.columns.str.strip()
        
        date_col = 'Date & Time' if 'Date & Time' in economic_events.columns else 'Date'
        economic_events['Date'] = pd.to_datetime(economic_events[date_col]).dt.date
        
        return economic_events, industry_stocks
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --------------------- CORE FUNCTIONS ---------------------
def calculate_risk_score(answers):
    score_map = {
        'tolerance': {'Low': 1, 'Medium': 3, 'High': 5},
        'horizon': {'<1 year': 1, '1-3 years': 2, '3-5 years': 3, '5+ years': 4},
        'experience': {'None': 1, 'Some': 3, 'Experienced': 5},
        'reaction': {'Sell all': 1, 'Sell some': 2, 'Hold': 3, 'Buy more': 5}
    }

    score = sum(score_map[key][answers[key]] for key in score_map)
    
    # Age adjustment
    age = int(answers['age'])
    if age > 60:
        score = max(1, score - 2)
    elif age > 40:
        score = max(1, score - 1)

    return min(10, max(1, score))

@st.cache_data
def get_volatility_metrics(ticker):
    try:
        data = yf.Ticker(ticker).history(period='1y')
        if len(data) < 50:
            return None
        
        daily_returns = data['Close'].pct_change().dropna()
        
        return {
            'volatility_score': min(10, int(np.std(daily_returns) * 100)),
            'max_drawdown': (data['Close'].max() - data['Close'].min()) / data['Close'].max(),
            'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) != 0 else 0,
            'beta': calculate_beta(ticker),
            'avg_daily_range': ((data['High'] - data['Low']).mean() / data['Close'].mean()) * 100
        }
    except Exception as e:
        st.warning(f"Volatility error for {ticker}: {e}")
        return None

def calculate_beta(ticker, benchmark='^GSPC', lookback=252):
    try:
        stock_data = yf.Ticker(ticker).history(period=f'{lookback}d')['Close']
        bench_data = yf.Ticker(benchmark).history(period=f'{lookback}d')['Close']
        
        merged = pd.concat([stock_data, bench_data], axis=1).pct_change().dropna()
        if len(merged) < 10:
            return 1.0
            
        cov_matrix = np.cov(merged.iloc[:,0], merged.iloc[:,1])
        return cov_matrix[0, 1] / cov_matrix[1, 1]
    except:
        return 1.0

@st.cache_data
def predict_stock_performance(ticker, horizon_days=30):
    try:
        data = yf.Ticker(ticker).history(period='2y')
        if len(data) < 100:
            return None
            
        # Feature engineering
        data['MA_50'] = data['Close'].rolling(50).mean()
        data['MA_200'] = data['Close'].rolling(200).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(20).std()
        data.dropna(inplace=True)
        
        # Prepare data
        X = data[['MA_50', 'MA_200', 'Volatility']]
        y = data['Close'].shift(-horizon_days).dropna()
        X = X.iloc[:-horizon_days]
        
        # Model training
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Prediction
        current_price = data['Close'].iloc[-1]
        predicted_price = model.predict(X.iloc[-1:].values)[0]
        predicted_change = (predicted_price - current_price) / current_price
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change': predicted_change,
            'confidence': min(95, int(70 + abs(predicted_change)*100))
        }
    except Exception as e:
        st.warning(f"Prediction error for {ticker}: {e}")
        return None

def generate_stock_plot(ticker):
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        data = yf.Ticker(ticker).history(period='6mo')
        ax.plot(data.index, data['Close'])
        ax.set_title(f'{ticker} 6-Month Price Trend')
        ax.set_ylabel('Price (₹)')
        ax.grid(True)
        return fig
    except:
        return None

# --------------------- UI COMPONENTS ---------------------
def show_risk_questionnaire():
    st.title("📊 Risk Profile Assessment")
    
    with st.form("risk_form"):
        st.write("#### Personal Information")
        age = st.number_input("Your Age", min_value=18, max_value=100, value=30)
        
        st.write("#### Investment Preferences")
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        
        investment_horizon = st.radio(
            "Investment Horizon",
            ["<1 year", "1-3 years", "3-5 years", "5+ years"],
            index=2
        )
        
        investment_experience = st.selectbox(
            "Investment Experience",
            ["None", "Some", "Experienced"],
            index=1
        )
        
        market_drop_reaction = st.radio(
            "If your portfolio loses 20% in a month, you would:",
            ["Sell all", "Sell some", "Hold", "Buy more"],
            index=2
        )
        
        if st.form_submit_button("Calculate Risk Score"):
            answers = {
                'age': age,
                'risk_tolerance': risk_tolerance,
                'investment_horizon': investment_horizon,
                'investment_experience': investment_experience,
                'market_drop_reaction': market_drop_reaction
            }
            st.session_state.risk_score = calculate_risk_score(answers)
            st.success(f"Your risk score: {st.session_state.risk_score}/10")
            
            risk_level = "Conservative" if st.session_state.risk_score < 4 else \
                       "Moderate" if st.session_state.risk_score < 7 else "Aggressive"
            st.info(f"Recommended strategy: **{risk_level}** portfolio allocation")

def show_dashboard():
    economic_events, industry_stocks = load_data()
    
    if economic_events.empty or industry_stocks.empty:
        st.error("Data loading failed. Check Excel files.")
        return
    
    st.title("📈 Eventfolio Dashboard")
    
    # Risk profile display
    if 'risk_score' not in st.session_state:
        st.warning("Complete the Risk Questionnaire first")
        if st.button("Go to Questionnaire"):
            st.session_state.current_page = "Risk Questionnaire"
            st.experimental_rerun()
        return
    
    risk_level = "Conservative" if st.session_state.risk_score < 4 else \
                "Moderate" if st.session_state.risk_score < 7 else "Aggressive"
    
    with st.sidebar:
        st.metric("Your Risk Profile", 
                 f"{risk_level} ({st.session_state.risk_score}/10)")
        
    # Upcoming events
    st.header("📅 Upcoming Economic Events")
    today = datetime.now().date()
    upcoming = economic_events[economic_events['Date'] >= today].sort_values('Date').head(10)
    
    if upcoming.empty:
        st.info("No upcoming events found")
    else:
        for _, event in upcoming.iterrows():
            with st.expander(f"{event['Economic Event']} - {event['Date']}"):
                col1, col2 = st.columns([3, 1])
                col1.write(f"**Country:** {event['Country']}")
                col2.write(f"**Impact:** {event['Impact']}")
                st.write(event.get('Description', 'No description available'))
                
                if st.button("Analyze Impact", key=f"analyze_{event['Economic Event']}"):
                    analyze_event(event['Economic Event'], industry_stocks)

def analyze_event(event_name, industry_stocks):
    st.title(f"🔍 {event_name} Analysis")
    
    related_stocks = industry_stocks[industry_stocks['Economic Event'] == event_name]
    if related_stocks.empty:
        st.warning("No stocks found for this event")
        return
    
    # Get all unique tickers
    tickers = []
    for stocks in related_stocks['Stocks']:
        if pd.notna(stocks):
            tickers.extend([t.strip() for t in stocks.split(',') if t.strip()])
    
    if not tickers:
        st.warning("No valid stock tickers found")
        return
    
    # Filter by risk
    filtered_stocks = []
    for ticker in tickers:
        metrics = get_volatility_metrics(ticker)
        if metrics:
            if st.session_state.risk_score < 4:  # Conservative
                if metrics['volatility_score'] < 4 and metrics['beta'] < 0.8:
                    filtered_stocks.append((ticker, metrics))
            elif st.session_state.risk_score < 7:  # Moderate
                if metrics['volatility_score'] < 6 and metrics['beta'] < 1.2:
                    filtered_stocks.append((ticker, metrics))
            else:  # Aggressive
                filtered_stocks.append((ticker, metrics))
    
    if not filtered_stocks:
        st.warning("No stocks match your risk profile")
        return
    
    st.header("📊 Recommended Stocks")
    st.write(f"Showing {len(filtered_stocks)} stocks filtered for your risk profile")
    
    for ticker, metrics in filtered_stocks:
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            
            # Header with ticker and predicted change
            prediction = predict_stock_performance(ticker)
            if not prediction:
                continue
                
            direction = 'up' if prediction['predicted_change'] > 0 else 'down'
            col1.subheader(ticker)
            col2.metric(
                "Predicted Change", 
                f"{prediction['predicted_change']*100:.1f}%",
                delta_color="normal" if direction == 'up' else "inverse"
            )
            
            # Metrics
            cols = st.columns(4)
            cols[0].metric("Volatility", f"{metrics['volatility_score']}/10")
            cols[1].metric("Beta", f"{metrics['beta']:.2f}")
            cols[2].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            cols[3].metric("Max Drawdown", f"{metrics['max_drawdown']*100:.1f}%")
            
            # Confidence
            st.progress(
                prediction['confidence']/100, 
                f"Confidence: {prediction['confidence']}%"
            )
            
            # Trading recommendation
            if prediction['predicted_change'] > 0.05:
                rec = "BUY"
                color = "green"
            elif prediction['predicted_change'] < -0.03:
                rec = "SELL"
                color = "red"
            else:
                rec = "HOLD"
                color = "orange"
            
            st.markdown(
                f"<div style='background-color:{color}20; padding:1rem; border-radius:8px; border-left:4px solid {color}'>"
                f"<h3 style='color:{color};text-align:center'>{rec} RECOMMENDATION</h3>"
                "</div>",
                unsafe_allow_html=True
            )
            
            # Price plot
            fig = generate_stock_plot(ticker)
            if fig:
                st.pyplot(fig)

# --------------------- MAIN APP ---------------------
def main():
    if 'risk_score' not in st.session_state:
        st.session_state.risk_score = None
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Navigation
    if st.session_state.current_page == "Risk Questionnaire":
        show_risk_questionnaire()
        if st.button("Back to Dashboard"):
            st.session_state.current_page = "Dashboard"
            st.experimental_rerun()
    else:
        show_dashboard()
        if st.sidebar.button("Update Risk Profile"):
            st.session_state.current_page = "Risk Questionnaire"
            st.experimental_rerun()

if __name__ == '__main__':
    main()
