#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1. First Streamlit command - must be at the very top
import streamlit as st
st.set_page_config(
    page_title="Eventfolio - Stock Event Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Standard library imports
import base64
from datetime import datetime, timedelta
from io import BytesIO
import warnings

# 3. Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# --------------------- DATA LOADING ---------------------
@st.cache_data
def load_data():
    """Load and preprocess the Excel data files"""
    try:
        economic_events = pd.read_excel('Economic Event.xlsx', engine='openpyxl')
        industry_stocks = pd.read_excel('Industry_EconomicEvent_Stocks.xlsx', engine='openpyxl')
        
        # Clean column names and handle dates
        economic_events.columns = economic_events.columns.str.strip()
        industry_stocks.columns = industry_stocks.columns.str.strip()
        
        date_col = 'Date & Time' if 'Date & Time' in economic_events.columns else 'Date'
        economic_events['Date'] = pd.to_datetime(economic_events[date_col]).dt.date
        
        return economic_events, industry_stocks
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# --------------------- CORE ANALYSIS FUNCTIONS ---------------------
def calculate_risk_score(answers):
    """Calculate user's risk score based on questionnaire answers"""
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period='1y'):
    """Get stock data from Yahoo Finance with error handling"""
    try:
        data = yf.Ticker(ticker).history(period=period)
        return data if not data.empty else None
    except Exception as e:
        st.warning(f"Failed to fetch data for {ticker}: {str(e)}")
        return None

def calculate_volatility_metrics(data):
    """Calculate volatility metrics from stock data"""
    if data is None or len(data) < 50:
        return None
    
    daily_returns = data['Close'].pct_change().dropna()
    metrics = {
        'volatility_score': min(10, int(np.std(daily_returns) * 100)),
        'max_drawdown': (data['Close'].max() - data['Close'].min()) / data['Close'].max(),
        'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) != 0 else 0,
        'avg_daily_range': ((data['High'] - data['Low']).mean() / data['Close'].mean()) * 100
    }
    return metrics

def calculate_beta(ticker, benchmark='^GSPC', lookback=252):
    """Calculate beta against a benchmark index"""
    try:
        stock_data = get_stock_data(ticker, f'{lookback}d')
        bench_data = get_stock_data(benchmark, f'{lookback}d')
        
        if stock_data is None or bench_data is None:
            return 1.0  # Default neutral beta
        
        merged = pd.concat([stock_data['Close'], bench_data['Close']], axis=1)
        merged.columns = ['stock', 'benchmark']
        merged = merged.pct_change().dropna()
        
        if len(merged) < 10:
            return 1.0
            
        cov_matrix = np.cov(merged['stock'], merged['benchmark'])
        return cov_matrix[0, 1] / cov_matrix[1, 1]
    except Exception:
        return 1.0

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def predict_stock_performance(ticker, horizon_days=30):
    """Predict future stock performance using ML and time series"""
    try:
        data = get_stock_data(ticker, '2y')
        if data is None or len(data) < 100:
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
        
        # Time series prediction
        ts_model = ARIMA(data['Close'].values, order=(5,1,0))
        ts_fit = ts_model.fit()
        ts_pred = ts_fit.forecast(steps=horizon_days)[-1]
        ts_change = (ts_pred - current_price) / current_price
        
        # Combined prediction
        combined_change = (predicted_change * 0.7 + ts_change * 0.3)
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change': combined_change,
            'confidence': min(95, int(70 + abs(combined_change)*100))
        }
    except Exception as e:
        st.warning(f"Prediction error for {ticker}: {str(e)}")
        return None

def generate_stock_chart(ticker, period='6mo'):
    """Generate price chart for a stock"""
    try:
        data = get_stock_data(ticker, period)
        if data is None:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data['Close'], linewidth=2)
        ax.set_title(f'{ticker} Price Trend ({period})', fontsize=12)
        ax.set_ylabel('Price (₹)', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception:
        return None

def generate_recommendation(prediction, metrics):
    """Generate trading recommendation with targets"""
    if prediction is None or metrics is None:
        return None
        
    current = prediction['current_price']
    volatility = metrics['volatility_score'] / 10
    beta = metrics.get('beta', 1.0)
    change = prediction['predicted_change']
    
    # Base recommendation
    if change > 0.05:
        rec = "BUY"
        target = min(prediction['predicted_price'], current * (1 + 0.1 + volatility * 0.5))
        stop_loss = current * (1 - max(0.03, 0.1 - volatility * 0.05))
    elif change < -0.03:
        rec = "SELL"
        target = max(prediction['predicted_price'], current * (1 - 0.1 - volatility * 0.5))
        stop_loss = current * (1 + max(0.03, 0.1 - volatility * 0.05))
    else:
        rec = "HOLD"
        target = current * 1.03
        stop_loss = current * 0.97
    
    # Adjust for beta
    if beta > 1.2:  # High beta stocks
        if rec == "BUY":
            target *= 1.05
            stop_loss *= 0.98
        elif rec == "SELL":
            target *= 0.95
            stop_loss *= 1.02
    
    # Calculate percentages
    target_pct = round((target - current) / current * 100, 1)
    stop_pct = round(abs(current - stop_loss) / current * 100, 1)
    reward_risk = round((target - current) / abs(current - stop_loss), 2)
    
    return {
        'action': rec,
        'target_price': round(target, 2),
        'stop_loss': round(stop_loss, 2),
        'target_pct': target_pct,
        'stop_pct': stop_pct,
        'reward_risk': reward_risk,
        'confidence': prediction['confidence']
    }

# --------------------- UI COMPONENTS ---------------------
def risk_questionnaire():
    """Display risk assessment questionnaire"""
    st.title("📝 Risk Profile Assessment")
    
    with st.form("risk_form"):
        st.write("### Personal Information")
        age = st.number_input("Your Age", min_value=18, max_value=100, value=35)
        
        st.write("### Investment Preferences")
        cols = st.columns(2)
        risk_tolerance = cols[0].select_slider(
            "Risk Tolerance",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        horizon = cols[1].selectbox(
            "Investment Horizon",
            options=["<1 year", "1-3 years", "3-5 years", "5+ years"],
            index=2
        )
        
        experience = st.radio(
            "Investment Experience Level",
            options=["Beginner", "Intermediate", "Advanced"],
            index=1
        )
        
        reaction = st.selectbox(
            "If your portfolio loses 20% in a month, you would:",
            options=["Sell all investments", "Sell some investments", 
                   "Hold and wait for recovery", "Buy more at lower prices"],
            index=2
        )
        
        if st.form_submit_button("Calculate My Risk Profile"):
            # Map inputs to scoring system
            exp_map = {"Beginner": "None", "Intermediate": "Some", "Advanced": "Experienced"}
            react_map = {
                "Sell all investments": "Sell all",
                "Sell some investments": "Sell some",
                "Hold and wait for recovery": "Hold",
                "Buy more at lower prices": "Buy more"
            }
            
            answers = {
                'age': age,
                'risk_tolerance': risk_tolerance,
                'investment_horizon': horizon,
                'investment_experience': exp_map[experience],
                'market_drop_reaction': react_map[reaction]
            }
            
            st.session_state.risk_score = calculate_risk_score(answers)
            st.session_state.risk_answers = answers
            st.success("Risk assessment completed!")
            st.experimental_rerun()

def display_risk_profile():
    """Show user's risk profile summary"""
    if 'risk_score' not in st.session_state:
        return
        
    score = st.session_state.risk_score
    risk_level = "Conservative" if score < 4 else "Moderate" if score < 7 else "Aggressive"
    
    with st.sidebar:
        st.subheader("Your Risk Profile")
        st.metric("Risk Score", f"{score}/10", delta=risk_level)
        
        # Display recommendation
        if score < 4:
            st.info("""
            **Recommended Portfolio:**  
            • 60% Bonds  
            • 30% Large-Cap Stocks  
            • 10% Cash
            """)
        elif score < 7:
            st.info("""
            **Recommended Portfolio:**  
            • 40% Large-Cap Stocks  
            • 30% Mid-Cap Stocks  
            • 20% Bonds  
            • 10% International
            """)
        else:
            st.info("""
            **Recommended Portfolio:**  
            • 50% Growth Stocks  
            • 20% Small-Cap  
            • 20% International  
            • 10% Alternative Investments
            """)

def event_dashboard():
    """Main dashboard showing upcoming events"""
    st.title("📈 Eventfolio Dashboard")
    economic_events, industry_stocks = load_data()
    
    if economic_events.empty or industry_stocks.empty:
        st.error("Failed to load required data files")
        return
    
    display_risk_profile()
    
    # Upcoming events section
    st.header("📅 Upcoming Economic Events")
    today = datetime.now().date()
    upcoming = economic_events[economic_events['Date'] >= today].sort_values('Date').head(10)
    
    if upcoming.empty:
        st.info("No upcoming economic events found")
    else:
        for _, event in upcoming.iterrows():
            with st.expander(f"{event['Economic Event']} - {event['Date']}"):
                cols = st.columns([3, 1])
                cols[0].write(f"**Country:** {event['Country']}")
                cols[0].write(f"**Impact:** {event['Impact']}")
                cols[1].write(event.get('Description', 'No description available'))
                
                if st.button("Analyze Impact", key=f"analyze_{event['Economic Event']}"):
                    analyze_event_stocks(event['Economic Event'], industry_stocks)

def analyze_event_stocks(event_name, industry_stocks):
    """Analyze stocks related to a specific economic event"""
    st.title(f"🔍 {event_name} Analysis")
    st.write("Analyzing stocks impacted by this event...")
    
    # Get all tickers for this event
    related = industry_stocks[industry_stocks['Economic Event'] == event_name]
    if related.empty:
        st.warning("No stock data available for this event")
        return
    
    tickers = []
    for stocks in related['Stocks']:
        if pd.notna(stocks):
            tickers.extend([t.strip() for t in stocks.split(',') if t.strip()])
    
    if not tickers:
        st.warning("No valid stock tickers found")
        return
    
    # Filter stocks based on risk profile
    filtered_stocks = []
    for ticker in tickers:
        data = get_stock_data(ticker)
        if data is None:
            continue
            
        metrics = calculate_volatility_metrics(data)
        if metrics is None:
            continue
            
        metrics['beta'] = calculate_beta(ticker)
        
        # Risk-based filtering
        score = st.session_state.get('risk_score', 5)
        if score < 4:  # Conservative
            if metrics['volatility_score'] < 4 and metrics['beta'] < 0.8:
                filtered_stocks.append((ticker, metrics))
        elif score < 7:  # Moderate
            if metrics['volatility_score'] < 6 and metrics['beta'] < 1.2:
                filtered_stocks.append((ticker, metrics))
        else:  # Aggressive
            filtered_stocks.append((ticker, metrics))
    
    if not filtered_stocks:
        st.warning("No stocks match your risk profile")
        return
    
    # Display analysis for each stock
    st.header("📊 Recommended Stocks")
    st.write(f"Found {len(filtered_stocks)} stocks matching your risk profile")
    
    for ticker, metrics in filtered_stocks:
        with st.container():
            st.markdown("---")
            
            # Get predictions
            prediction = predict_stock_performance(ticker)
            if prediction is None:
                continue
                
            recommendation = generate_recommendation(prediction, metrics)
            if recommendation is None:
                continue
            
            # Stock header
            cols = st.columns([3, 1])
            cols[0].subheader(ticker)
            
            # Price change indicator
            change = prediction['predicted_change']
            delta_color = "normal" if change > 0 else "inverse"
            cols[1].metric(
                "Predicted Change", 
                f"{change*100:.1f}%",
                delta_color=delta_color
            )
            
            # Key metrics
            m_cols = st.columns(4)
            m_cols[0].metric("Volatility", f"{metrics['volatility_score']}/10")
            m_cols[1].metric("Beta", f"{metrics['beta']:.2f}")
            m_cols[2].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            m_cols[3].metric("Max Drawdown", f"{metrics['max_drawdown']*100:.1f}%")
            
            # Confidence indicator
            st.progress(
                recommendation['confidence']/100, 
                f"Analysis Confidence: {recommendation['confidence']}%"
            )
            
            # Trading recommendation
            rec_color = {
                "BUY": "green",
                "SELL": "red",
                "HOLD": "orange"
            }[recommendation['action']]
            
            st.markdown(
                f"""
                <div style='background-color:{rec_color}10; 
                    padding:1rem; 
                    border-radius:8px; 
                    border-left:4px solid {rec_color};
                    margin:1rem 0;'>
                    <h3 style='color:{rec_color}; text-align:center;'>
                        {recommendation['action']} RECOMMENDATION
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Price targets
            t_cols = st.columns(3)
            t_cols[0].metric("Current Price", f"₹{prediction['current_price']:.2f}")
            t_cols[1].metric(
                "Target Price", 
                f"₹{recommendation['target_price']:.2f}", 
                f"+{recommendation['target_pct']}%"
            )
            t_cols[2].metric(
                "Stop Loss", 
                f"₹{recommendation['stop_loss']:.2f}", 
                f"-{recommendation['stop_pct']}%",
                delta_color="inverse"
            )
            
            # Risk/reward
            st.metric("Risk/Reward Ratio", f"{recommendation['reward_risk']}:1")
            
            # Price chart
            chart = generate_stock_chart(ticker)
            if chart:
                st.pyplot(chart)

# --------------------- MAIN APP ---------------------
def main():
    """Main application controller"""
    # Initialize session state
    if 'risk_score' not in st.session_state:
        st.session_state.risk_score = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Navigation sidebar
    with st.sidebar:
        st.title("Eventfolio")
        
        if st.session_state.risk_score:
            display_risk_profile()
        
        if st.button("🔄 Update Risk Profile"):
            st.session_state.current_page = "Questionnaire"
            st.experimental_rerun()
        
        st.markdown("---")
        st.write("Navigate to:")
        if st.button("📊 Dashboard"):
            st.session_state.current_page = "Dashboard"
            st.experimental_rerun()
    
    # Page routing
    if st.session_state.current_page == "Questionnaire":
        risk_questionnaire()
        if st.session_state.risk_score is not None:
            if st.button("← Back to Dashboard"):
                st.session_state.current_page = "Dashboard"
                st.experimental_rerun()
    else:
        event_dashboard()

if __name__ == '__main__':
    main()
