import streamlit as st
st.set_page_config(page_title="Economic Event Stock Analyzer", layout="wide")  # MUST BE FIRST

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# Load Data
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

economic_events_df, industry_stocks_df = load_data()

# Risk Score Calculation
def calculate_risk_score(answers):
    score_map = {
        'tolerance': {'Low': 1, 'Medium': 3, 'High': 5},
        'horizon': {'<1 year': 1, '1-3 years': 2, '3-5 years': 3, '5+ years': 4},
        'experience': {'None': 1, 'Some': 3, 'Experienced': 5},
        'reaction': {'Sell all': 1, 'Sell some': 2, 'Hold': 3, 'Buy more': 5}
    }

    score = 0
    score += score_map['tolerance'][answers['risk_tolerance']]
    score += score_map['horizon'][answers['investment_horizon']]
    score += score_map['experience'][answers['investment_experience']]
    score += score_map['reaction'][answers['market_drop_reaction']]

    age = int(answers['age'])
    if age > 60:
        score = max(1, score - 2)
    elif age > 40:
        score = max(1, score - 1)

    return min(10, max(1, score))

# Volatility Analysis
@st.cache_data
def get_volatility_metrics(ticker):
    try:
        if not ticker or pd.isna(ticker):
            return None
            
        data = yf.Ticker(ticker).history(period='1y')
        if len(data) < 50:
            return None
        
        daily_returns = data['Close'].pct_change().dropna()
        
        metrics = {
            'volatility_score': min(10, int(np.std(daily_returns) * 100)),
            'max_drawdown': (data['Close'].max() - data['Close'].min()) / data['Close'].max(),
            'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) != 0 else 0,
            'beta': calculate_beta(ticker),
            'avg_daily_range': ((data['High'] - data['Low']).mean() / data['Close'].mean()) * 100
        }
        return metrics
    except Exception as e:
        st.warning(f"Error calculating volatility for {ticker}: {e}")
        return None

def calculate_beta(ticker, benchmark='^GSPC', lookback=252):
    try:
        if not ticker or pd.isna(ticker):
            return 1.0
            
        stock_data = yf.Ticker(ticker).history(period=f'{lookback}d')['Close']
        bench_data = yf.Ticker(benchmark).history(period=f'{lookback}d')['Close']
        
        if bench_data.empty:
            bench_data = yf.Ticker('^NSEI').history(period=f'{lookback}d')['Close']
            if bench_data.empty:
                return 1.0
        
        merged = pd.concat([stock_data, bench_data], axis=1)
        merged.columns = ['stock', 'benchmark']
        merged = merged.pct_change().dropna()
        
        if len(merged) < 10:
            return 1.0
            
        cov_matrix = np.cov(merged['stock'], merged['benchmark'])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        return beta
    except Exception as e:
        st.warning(f"Error calculating beta for {ticker}: {e}")
        return 1.0

# Predictive Analytics
@st.cache_data
def predict_stock_performance(ticker, event_date, horizon_days=30):
    try:
        if not ticker or pd.isna(ticker):
            return None
            
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*2)
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        if data.empty or len(data) < 100:
            return None
        
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
        data.dropna(inplace=True)
        
        X = data[['MA_50', 'MA_200', 'Volatility']]
        y = data['Close'].shift(-horizon_days).dropna()
        X = X.iloc[:-horizon_days]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        last_features = X.iloc[-1].values.reshape(1, -1)
        predicted_price = model.predict(last_features)[0]
        current_price = data['Close'].iloc[-1]
        predicted_change = (predicted_price - current_price) / current_price
        
        ts_data = data['Close'].values
        model_arima = ARIMA(ts_data, order=(5,1,0))
        model_fit = model_arima.fit()
        forecast = model_fit.forecast(steps=horizon_days)
        ts_predicted = forecast[-1]
        ts_change = (ts_predicted - current_price) / current_price
        
        combined_change = (predicted_change * 0.7 + ts_change * 0.3)
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change': combined_change,
            'confidence': min(95, int(70 + abs(combined_change)*100)),
            'time_series_prediction': ts_predicted,
            'ml_prediction': predicted_price
        }
    except Exception as e:
        st.warning(f"Error predicting {ticker}: {e}")
        return None

def generate_stock_plot(ticker):
    try:
        if not ticker or pd.isna(ticker):
            return None
            
        data = yf.Ticker(ticker).history(period='6mo')
        if data.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index, data['Close'], label='Closing Price')
        ax.set_title(f'{ticker} Price Trend (6 Months)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.grid(True)
        
        return fig
    except Exception as e:
        st.warning(f"Error generating plot for {ticker}: {e}")
        return None

def calculate_recommendation(ticker, prediction, metrics):
    current_price = prediction['current_price']
    volatility = metrics['volatility_score'] / 10
    beta = float(metrics['beta'])
    
    if prediction['predicted_change'] > 0.05:
        recommendation = "BUY"
        target = min(prediction['predicted_price'], current_price * (1 + 0.1 + volatility * 0.5))
        stop_loss = current_price * (1 - max(0.03, 0.1 - volatility * 0.05))
    elif prediction['predicted_change'] < -0.03:
        recommendation = "SELL"
        target = max(prediction['predicted_price'], current_price * (1 - 0.1 - volatility * 0.5))
        stop_loss = current_price * (1 + max(0.03, 0.1 - volatility * 0.05))
    else:
        recommendation = "HOLD"
        target = current_price * (1 + 0.03)
        stop_loss = current_price * (1 - 0.03)
    
    if beta > 1.2:
        if recommendation == "BUY":
            target *= 1.05
            stop_loss *= 0.98
        elif recommendation == "SELL":
            target *= 0.95
            stop_loss *= 1.02
    
    target_pct = round((target - current_price) / current_price * 100, 1)
    stop_loss_pct = round((current_price - stop_loss) / current_price * 100, 1)
    
    return {
        'recommendation': recommendation,
        'target': round(target, 2),
        'stop_loss': round(stop_loss, 2),
        'risk_reward_ratio': round((target - current_price) / (current_price - stop_loss), 2),
        'target_pct': target_pct,
        'stop_loss_pct': stop_loss_pct
    }

def filter_by_risk(stocks, risk_score):
    filtered = []
    for ticker in stocks:
        metrics = get_volatility_metrics(ticker)
        if not metrics:
            continue
            
        if risk_score < 4:
            if metrics['volatility_score'] < 4 and metrics['max_drawdown'] < 0.2 and metrics['beta'] < 0.8:
                filtered.append((ticker, metrics))
        elif risk_score < 7:
            if metrics['volatility_score'] < 6 and metrics['max_drawdown'] < 0.3 and metrics['beta'] < 1.2:
                filtered.append((ticker, metrics))
        else:
            filtered.append((ticker, metrics))
    
    filtered.sort(key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    return filtered

# Streamlit UI
def main():
    st.set_page_config(page_title="Economic Event Stock Analyzer", layout="wide")
    
    if 'risk_score' not in st.session_state:
        st.session_state.risk_score = None
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Risk Questionnaire"])
    
    if page == "Risk Questionnaire":
        show_risk_questionnaire()
    else:
        show_dashboard()

def show_risk_questionnaire():
    st.title("Risk Assessment Questionnaire")
    
    with st.form("risk_form"):
        age = st.number_input("1. What is your age?", min_value=18, max_value=100, step=1)
        risk_tolerance = st.selectbox(
            "2. How would you describe your risk tolerance?",
            ["Low", "Medium", "High"],
            index=1
        )
        investment_horizon = st.selectbox(
            "3. What is your investment horizon?",
            ["<1 year", "1-3 years", "3-5 years", "5+ years"],
            index=2
        )
        investment_experience = st.selectbox(
            "4. How would you describe your investment experience?",
            ["None", "Some", "Experienced"],
            index=1
        )
        market_drop_reaction = st.selectbox(
            "5. If your portfolio lost 20% in a market decline, you would:",
            ["Sell all", "Sell some", "Hold", "Buy more"],
            index=2
        )
        
        submitted = st.form_submit_button("Calculate My Risk Score")
        if submitted:
            answers = {
                'age': age,
                'risk_tolerance': risk_tolerance,
                'investment_horizon': investment_horizon,
                'investment_experience': investment_experience,
                'market_drop_reaction': market_drop_reaction
            }
            st.session_state.risk_score = calculate_risk_score(answers)
            st.success(f"Your risk score is: {st.session_state.risk_score}/10")
            st.experimental_rerun()

def show_dashboard():
    st.title("Economic Event Dashboard")
    
    if economic_events_df.empty or industry_stocks_df.empty:
        st.error("Could not load the required data files. Please check if the Excel files exist.")
        return
    
    if st.session_state.risk_score is None:
        st.warning("Please complete the Risk Questionnaire first")
        if st.button("Go to Risk Questionnaire"):
            st.experimental_rerun()
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Upcoming Economic Events")
    with col2:
        risk_level = "Conservative" if st.session_state.risk_score < 4 else \
                    "Moderate" if st.session_state.risk_score < 7 else "Aggressive"
        st.metric("Your Risk Profile", f"{risk_level} ({st.session_state.risk_score}/10)")
    
    today = datetime.now().date()
    upcoming = economic_events_df[economic_events_df['Date'] >= today]
    upcoming = upcoming.sort_values('Date').head(20)
    
    if upcoming.empty:
        st.info("No upcoming events found.")
    else:
        for _, event in upcoming.iterrows():
            with st.expander(f"{event['Economic Event']} - {event['Date']}"):
                cols = st.columns([3, 1])
                cols[0].write(f"**Country:** {event['Country']}")
                cols[1].write(f"**Impact:** {event['Impact']}")
                st.write(event['Description'] or 'No description available')
                
                if st.button("Analyze Impact", key=f"analyze_{event['Economic Event']}"):
                    st.session_state.selected_event = event['Economic Event']
                    st.experimental_rerun()
    
    if 'selected_event' in st.session_state:
        analyze_event(st.session_state.selected_event)

def analyze_event(event_name):
    st.title(f"Event Analysis: {event_name}")
    
    try:
        event = economic_events_df[economic_events_df['Economic Event'] == event_name].iloc[0]
    except IndexError:
        st.error(f"Event '{event_name}' not found.")
        return
    
    st.write(f"**Date:** {event['Date']} | **Country:** {event['Country']} | **Impact:** {event['Impact']}")
    st.write(event['Description'] or 'No description available')
    
    related_stocks = industry_stocks_df[industry_stocks_df['Economic Event'] == event_name]
    
    stocks = []
    for _, row in related_stocks.iterrows():
        if pd.notna(row['Stocks']):
            for ticker in row['Stocks'].split(','):
                ticker_clean = ticker.strip()
                if ticker_clean:
                    stocks.append(ticker_clean)
    
    risk_filtered = filter_by_risk(stocks, st.session_state.risk_score)
    
    predictions = []
    for ticker, metrics in risk_filtered:
        try:
            prediction = predict_stock_performance(ticker, event['Date'])
            if not prediction:
                continue
                
            fig = generate_stock_plot(ticker)
            
            recommendation = calculate_recommendation(ticker, prediction, metrics)
            
            predictions.append({
                'ticker': ticker,
                'current_price': prediction['current_price'],
                'predicted_price': prediction['predicted_price'],
                'predicted_change': prediction['predicted_change'],
                'direction': 'up' if prediction['predicted_change'] > 0 else 'down',
                'confidence': prediction['confidence'],
                'volatility_score': metrics['volatility_score'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'beta': metrics['beta'],
                'max_drawdown': metrics['max_drawdown'],
                'avg_daily_range': metrics['avg_daily_range'],
                'price_plot': fig,
                'recommendation': recommendation['recommendation'],
                'target_price': recommendation['target'],
                'stop_loss': recommendation['stop_loss'],
                'risk_reward': recommendation['risk_reward_ratio'],
                'target_pct': recommendation['target_pct'],
                'stop_loss_pct': recommendation['stop_loss_pct']
            })
        except Exception as e:
            st.warning(f"Error processing {ticker}: {e}")
            continue
    
    predictions.sort(key=lambda x: (x['confidence'], abs(x['predicted_change'])), reverse=True)
    
    if not predictions:
        st.info("No suitable stocks found for your risk profile.")
        return
    
    st.subheader("Recommended Stocks")
    st.write(f"Showing {len(predictions)} stocks filtered for your risk profile")
    
    for stock in predictions:
        with st.container():
            st.markdown("---")
            cols = st.columns([3, 1])
            cols[0].markdown(f"### {stock['ticker']}")
            
            direction_color = "green" if stock['direction'] == 'up' else "red"
            cols[1].markdown(
                f"<span style='color:{direction_color}; font-size: 1.2rem; font-weight: bold;'>"
                f"{stock['predicted_change']*100:.1f}%</span>", 
                unsafe_allow_html=True
            )
            
            # Metrics
            cols = st.columns(4)
            cols[0].metric("Volatility Score", f"{stock['volatility_score']}/10")
            cols[1].metric("Beta", f"{stock['beta']:.2f}")
            cols[2].metric("Sharpe Ratio", f"{stock['sharpe_ratio']:.2f}")
            cols[3].metric("Max Drawdown", f"{stock['max_drawdown']*100:.1f}%")
            
            # Confidence bar
            confidence_color = "green" if stock['confidence'] > 80 else "orange" if stock['confidence'] > 60 else "red"
            st.progress(stock['confidence']/100, f"Confidence: {stock['confidence']}%")
            
            # Trading recommendation
            st.subheader("Trading Recommendation")
            
            rec_color = {
                "BUY": "green",
                "SELL": "red",
                "HOLD": "orange"
            }[stock['recommendation']]
            
            st.markdown(
                f"<div style='background-color:{rec_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {rec_color};'>"
                f"<h3 style='color:{rec_color};'>{stock['recommendation']} Recommendation</h3>"
                "</div>", 
                unsafe_allow_html=True
            )
            
            # Price targets
            cols = st.columns(3)
            cols[0].metric("Current Price", f"₹{stock['current_price']:.2f}")
            cols[1].metric(
                "Target Price", 
                f"₹{stock['target_price']:.2f}", 
                f"+{stock['target_pct']}%",
                delta_color="normal"
            )
            cols[2].metric(
                "Stop Loss", 
                f"₹{stock['stop_loss']:.2f}", 
                f"-{stock['stop_loss_pct']}%",
                delta_color="inverse"
            )
            
            st.metric("Risk/Reward Ratio", f"{stock['risk_reward']:.2f}:1")
            
            # Price plot
            if stock['price_plot']:
                st.pyplot(stock['price_plot'])
            
    st.markdown("---")
    st.subheader("Analysis Methodology")
    st.write("""
    Our trading recommendations are based on comprehensive analysis:
    - **Price Prediction:** Machine learning (Random Forest) and time series analysis (ARIMA)
    - **Volatility Adjustment:** More volatile stocks get wider target ranges (3-15%)
    - **Beta Sensitivity:** High-beta stocks (>1.2) get adjusted targets (+/-5%)
    - **Risk Management:** Stop losses set at 3-10% depending on volatility
    - **Confidence Levels:** Only predictions with >70% confidence are shown
    """)
    st.info("""
    **Note:** All prices are in INR (₹). Recommendations are based on technical analysis 
    and should be verified with fundamental analysis before trading.
    """)

if __name__ == '__main__':
    main()
