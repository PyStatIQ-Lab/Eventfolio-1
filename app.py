import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from fpdf import FPDF
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import base64
import warnings
warnings.filterwarnings('ignore')

# ========== MUST BE THE FIRST STREAMLIT COMMAND ==========
st.set_page_config(
    page_title="Investment Risk Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== NOW YOU CAN ADD OTHER STREAMLIT COMMANDS ==========
# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stSlider, .stTextInput {
        border-radius: 5px;
    }
    .metric-card {
        background-color: grey;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .risk-low {
        color: #28a745;
    }
    .risk-medium {
        color: #ffc107;
    }
    .risk-high {
        color: #dc3545;
    }
    .stock-card {
        background-color: grey;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .event-card {
        background-color: grey;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    try:
        economic_events = pd.read_excel('data/Economic Event.xlsx')
        industry_stocks = pd.read_excel('data/Industry_EconomicEvent_Stocks.xlsx')
        
        economic_events.columns = economic_events.columns.str.strip()
        industry_stocks.columns = industry_stocks.columns.str.strip()
        
        date_col = 'Date & Time' if 'Date & Time' in economic_events.columns else 'Date'
        economic_events['Date'] = pd.to_datetime(economic_events[date_col]).dt.date
        
        return economic_events, industry_stocks
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

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
        st.error(f"Error calculating volatility for {ticker}: {e}")
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
        st.error(f"Error calculating beta for {ticker}: {e}")
        return 1.0

def predict_stock_performance(ticker, event_date, horizon_days=30):
    try:
        if not ticker or pd.isna(ticker):
            return None
            
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365*2)
        data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        if data.empty or len(data) < 100:
            return None
        
        # Prepare features
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['MA_200'] = data['Close'].rolling(window=200).mean()
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
        data.dropna(inplace=True)
        
        # Machine Learning Model
        X = data[['MA_50', 'MA_200', 'Volatility']]
        y = data['Close'].shift(-horizon_days).dropna()
        X = X.iloc[:-horizon_days]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        last_features = X.iloc[-1].values.reshape(1, -1)
        predicted_price = model.predict(last_features)[0]
        current_price = data['Close'].iloc[-1]
        predicted_change = (predicted_price - current_price) / current_price
        
        # Time Series Model
        ts_data = data['Close'].values
        model_arima = ARIMA(ts_data, order=(5,1,0))
        model_fit = model_arima.fit()
        forecast = model_fit.forecast(steps=horizon_days)
        ts_predicted = forecast[-1]
        ts_change = (ts_predicted - current_price) / current_price
        
        # Combined prediction
        combined_change = (predicted_change * 0.7 + ts_change * 0.3)
        
        # Generate forecast data for visualization
        forecast_dates = pd.date_range(start=data.index[-1], periods=horizon_days+1)[1:]
        ml_forecast = [predicted_price] * horizon_days
        ts_forecast = forecast
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_change': combined_change,
            'confidence': min(95, int(70 + abs(combined_change)*100)),
            'time_series_prediction': ts_predicted,
            'ml_prediction': predicted_price,
            'forecast_dates': forecast_dates,
            'ml_forecast': ml_forecast,
            'ts_forecast': ts_forecast,
            'historical_data': data
        }
    except Exception as e:
        st.error(f"Error predicting {ticker}: {e}")
        return None

def generate_stock_chart(ticker, prediction_data):
    try:
        if not ticker or pd.isna(ticker):
            return None
            
        # Create figure with secondary y-axis for volume
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add historical price trace
        fig.add_trace(
            go.Scatter(
                x=prediction_data['historical_data'].index,
                y=prediction_data['historical_data']['Close'],
                name='Closing Price',
                line=dict(color='#1f77b4')
            ),
            secondary_y=False,
        )
        
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=prediction_data['historical_data'].index,
                y=prediction_data['historical_data']['MA_50'],
                name='50-Day MA',
                line=dict(color='#ff7f0e', dash='dot')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=prediction_data['historical_data'].index,
                y=prediction_data['historical_data']['MA_200'],
                name='200-Day MA',
                line=dict(color='#2ca02c', dash='dot')
            ),
            secondary_y=False,
        )
        
        # Add forecasted values
        fig.add_trace(
            go.Scatter(
                x=prediction_data['forecast_dates'],
                y=prediction_data['ml_forecast'],
                name='ML Forecast',
                line=dict(color='#d62728', dash='dash')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=prediction_data['forecast_dates'],
                y=prediction_data['ts_forecast'],
                name='TS Forecast',
                line=dict(color='#9467bd', dash='dash')
            ),
            secondary_y=False,
        )
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=prediction_data['historical_data'].index,
                y=prediction_data['historical_data']['Volume'],
                name='Volume',
                marker_color='#7f7f7f',
                opacity=0.5
            ),
            secondary_y=True,
        )
        
        # Add current price annotation
        fig.add_annotation(
            x=prediction_data['historical_data'].index[-1],
            y=prediction_data['current_price'],
            text=f"Current: ₹{prediction_data['current_price']:.2f}",
            showarrow=True,
            arrowhead=1
        )
        
        # Add predicted price annotation
        fig.add_annotation(
            x=prediction_data['forecast_dates'][-1],
            y=prediction_data['predicted_price'],
            text=f"Predicted: ₹{prediction_data['predicted_price']:.2f}",
            showarrow=True,
            arrowhead=1
        )
        
        # Set layout
        fig.update_layout(
            title=f'{ticker} Price Analysis with 30-Day Forecast',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        
        return fig
    except Exception as e:
        st.error(f"Error generating chart for {ticker}: {e}")
        return None

def calculate_recommendation(ticker, prediction, metrics):
    current_price = prediction['current_price']
    volatility = metrics['volatility_score'] / 10
    beta = float(metrics['beta'])
    
    if prediction['predicted_change'] > 0.05:
        recommendation = "BUY"
        target = min(
            prediction['predicted_price'],
            current_price * (1 + 0.1 + volatility * 0.5)
        )
        stop_loss = current_price * (1 - max(0.03, 0.1 - volatility * 0.05))
    elif prediction['predicted_change'] < -0.03:
        recommendation = "SELL"
        target = max(
            prediction['predicted_price'],
            current_price * (1 - 0.1 - volatility * 0.5)
        )
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
    
    return {
        'recommendation': recommendation,
        'target': round(target, 2),
        'stop_loss': round(stop_loss, 2),
        'risk_reward_ratio': round((target - current_price) / (current_price - stop_loss), 2)
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

def generate_pdf_report(risk_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Investment Risk Profile Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Risk Score: {risk_score}/10", ln=1)

    risk_level = "Conservative" if risk_score < 4 else \
                 "Moderate" if risk_score < 7 else "Aggressive"
    pdf.multi_cell(0, 10, txt=f"Your risk profile: {risk_level}")
    
    if risk_score < 4:
        pdf.multi_cell(0, 10, txt="Recommended Portfolio: 60% Bonds, 30% Large-Cap Stocks, 10% Cash")
    elif risk_score < 7:
        pdf.multi_cell(0, 10, txt="Recommended Portfolio: 40% Large-Cap Stocks, 30% Mid-Cap Stocks, 20% Bonds, 10% International")
    else:
        pdf.multi_cell(0, 10, txt="Recommended Portfolio: 50% Growth Stocks, 20% Small-Cap, 20% International, 10% Alternative Investments")

    return pdf.output(dest='S').encode('latin1')

def risk_questionnaire():
    st.title("📊 Risk Profile Questionnaire")
    
    with st.form("risk_form"):
        st.markdown("### Please answer the following questions to assess your risk tolerance:")
        
        age = st.selectbox("Your Age", ["<30", "30-40", "40-50", "50-60", ">60"])
        risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        investment_horizon = st.selectbox("Investment Horizon", ["<1 year", "1-3 years", "3-5 years", "5+ years"])
        investment_experience = st.selectbox("Investment Experience", ["None", "Some", "Experienced"])
        market_drop_reaction = st.selectbox("If your portfolio drops 20% in a month, you would:", 
                                          ["Sell all", "Sell some", "Hold", "Buy more"])
        
        submitted = st.form_submit_button("Calculate Risk Score")
        
        if submitted:
            answers = {
                'age': age,
                'risk_tolerance': risk_tolerance,
                'investment_horizon': investment_horizon,
                'investment_experience': investment_experience,
                'market_drop_reaction': market_drop_reaction
            }
            
            # Convert age to numeric value for calculation
            age_map = {"<30": 25, "30-40": 35, "40-50": 45, "50-60": 55, ">60": 65}
            answers['age'] = str(age_map[age])
            
            risk_score = calculate_risk_score(answers)
            st.session_state['risk_score'] = risk_score
            
            risk_level = "Conservative" if risk_score < 4 else \
                         "Moderate" if risk_score < 7 else "Aggressive"
            
            st.success(f"Your Risk Score: {risk_score}/10 ({risk_level})")
            
            # Display risk meter
            col1, col2, col3 = st.columns([1, 4, 1])
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(to right, #28a745 0%, #ffc107 50%, #dc3545 100%); 
                            height: 20px; border-radius: 10px; position: relative; margin: 15px 0;">
                    <div style="position: absolute; left: {risk_score*10}%; top: -5px; 
                                width: 10px; height: 30px; background: black; border-radius: 5px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: -10px;">
                    <span>Low Risk</span>
                    <span>High Risk</span>
                </div>
                """, unsafe_allow_html=True)
            
            if risk_score < 4:
                st.info("""
                **Recommended Strategy:**  
                - Focus on capital preservation  
                - Prefer low-volatility investments  
                - Consider bonds, large-cap stocks, and fixed income instruments
                """)
            elif risk_score < 7:
                st.info("""
                **Recommended Strategy:**  
                - Balanced approach between growth and safety  
                - Diversify across asset classes  
                - Mix of large-cap and mid-cap stocks with some bonds
                """)
            else:
                st.info("""
                **Recommended Strategy:**  
                - Focus on capital growth  
                - Can tolerate higher volatility  
                - Consider growth stocks, small-caps, and international exposure
                """)

def dashboard():
    st.title("📈 Economic Event Dashboard")
    
    if 'risk_score' not in st.session_state:
        st.warning("Please complete the risk questionnaire first.")
        risk_questionnaire()
        return
    
    economic_events_df, industry_stocks_df = load_data()
    
    if economic_events_df.empty or industry_stocks_df.empty:
        st.error("Failed to load data. Please check the data files.")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        risk_score = st.slider("Adjust Risk Tolerance", 1, 10, st.session_state['risk_score'])
        st.session_state['risk_score'] = risk_score
        
        st.markdown("---")
        st.markdown("### Quick Links")
        if st.button("Risk Questionnaire"):
            st.session_state['page'] = 'questionnaire'
        if st.button("Risk Report"):
            st.session_state['page'] = 'report'
        
        st.markdown("---")
        st.markdown("**Current Risk Profile**")
        risk_level = "Conservative" if risk_score < 4 else \
                     "Moderate" if risk_score < 7 else "Aggressive"
        risk_class = "risk-low" if risk_score < 4 else "risk-medium" if risk_score < 7 else "risk-high"
        st.markdown(f"<div class='metric-card'><h3 class='{risk_class}'>{risk_score}/10 ({risk_level})</h3></div>", unsafe_allow_html=True)
    
    # Dashboard metrics
    today = datetime.now().date()
    upcoming = economic_events_df[economic_events_df['Date'] >= today]
    upcoming = upcoming.sort_values('Date').head(20)
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Upcoming Events</h4><h2>{len(upcoming)}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>Your Risk Score</h4><h2 class='{risk_class}'>{risk_score}/10</h2></div>", unsafe_allow_html=True)
    with col3:
        filtered_stocks = industry_stocks_df[industry_stocks_df['Economic Event'].isin(upcoming['Economic Event'])]
        st.markdown(f"<div class='metric-card'><h4>Affected Stocks</h4><h2>{len(filtered_stocks['Stocks'].dropna().unique())}</h2></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader(f"📅 Upcoming Economic Events (Filtered for Risk Profile: {risk_score}/10)")
    
    for _, event in upcoming.iterrows():
        with st.expander(f"📌 {event['Date']} - {event['Economic Event']}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div class='event-card'>
                    <p><strong>Country:</strong> {event['Country']}</p>
                    <p><strong>Impact:</strong> {event.get('Impact', 'N/A')}</p>
                    <p><strong>Previous:</strong> {event.get('Previous', 'N/A')}</p>
                    <p><strong>Forecast:</strong> {event.get('Forecast', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"Analyze Impact on Stocks", key=f"analyze_{event['Date']}_{event['Economic Event']}"):
                    st.session_state['selected_event'] = event
                    st.session_state['page'] = 'analysis'

def analyze_event(event):
    st.title(f"🔍 Analysis: {event['Economic Event']}")
    
    # Back button
    if st.button("← Back to Dashboard"):
        st.session_state['page'] = 'dashboard'
        return
    
    # Event details
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class='event-card'>
            <h4>Event Details</h4>
            <p><strong>Date:</strong> {event['Date']}</p>
            <p><strong>Country:</strong> {event['Country']}</p>
            <p><strong>Impact:</strong> {event.get('Impact', 'N/A')}</p>
            <p><strong>Previous:</strong> {event.get('Previous', 'N/A')}</p>
            <p><strong>Forecast:</strong> {event.get('Forecast', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='event-card'>
            <h4>Risk Profile</h4>
            <p><strong>Your Risk Score:</strong> {st.session_state['risk_score']}/10</p>
            <p><strong>Filtered Stocks:</strong> Showing only stocks matching your risk tolerance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stock analysis
    _, industry_stocks_df = load_data()
    related_stocks = industry_stocks_df[industry_stocks_df['Economic Event'] == event['Economic Event']]

    stocks = []
    for _, row in related_stocks.iterrows():
        if pd.notna(row['Stocks']):
            for ticker in row['Stocks'].split(','):
                ticker_clean = ticker.strip()
                if ticker_clean:
                    stocks.append(ticker_clean)

    risk_filtered = filter_by_risk(stocks, st.session_state['risk_score'])

    if not risk_filtered:
        st.warning("No suitable stocks found for your risk profile.")
        return
    
    st.subheader(f"📊 Stock Recommendations (Filtered for Risk Score: {st.session_state['risk_score']})")
    
    for ticker, metrics in risk_filtered:
        prediction = predict_stock_performance(ticker, event['Date'])
        if not prediction:
            continue
            
        recommendation = calculate_recommendation(ticker, prediction, metrics)
        chart = generate_stock_chart(ticker, prediction)
        
        st.markdown(f"""
        <div class='stock-card'>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3>{ticker}</h3>
                <span style="font-size: 1.2rem; font-weight: bold; 
                    color: {'#dc3545' if recommendation['recommendation'] == 'SELL' else '#28a745'}">
                    {recommendation['recommendation']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{prediction['current_price']:.2f}")
        col2.metric("Predicted Change", f"{prediction['predicted_change']*100:.1f}%", 
                   delta_color="inverse" if prediction['predicted_change'] < 0 else "normal")
        col3.metric("Volatility", f"{metrics['volatility_score']}/10")
        col4.metric("Beta", f"{metrics['beta']:.2f}")
        
        # Display the interactive chart
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Detailed metrics
        with st.expander("📈 Detailed Analysis"):
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"""
            <div class='metric-card'>
                <h4>Performance Metrics</h4>
                <p>Sharpe Ratio: {metrics['sharpe_ratio']:.2f}</p>
                <p>Max Drawdown: {metrics['max_drawdown']*100:.1f}%</p>
                <p>Avg Daily Range: {metrics['avg_daily_range']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            col2.markdown(f"""
            <div class='metric-card'>
                <h4>Prediction Details</h4>
                <p>Confidence: {prediction['confidence']}%</p>
                <p>ML Prediction: ₹{prediction['ml_prediction']:.2f}</p>
                <p>TS Prediction: ₹{prediction['time_series_prediction']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col3.markdown(f"""
            <div class='metric-card'>
                <h4>Trading Strategy</h4>
                <p>Target Price: ₹{recommendation['target']:.2f}</p>
                <p>Stop Loss: ₹{recommendation['stop_loss']:.2f}</p>
                <p>Risk/Reward: {recommendation['risk_reward_ratio']:.2f}:1</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

def report_page():
    st.title("📑 Investment Risk Profile Report")
    
    if 'risk_score' not in st.session_state:
        st.warning("Please complete the risk questionnaire first.")
        risk_questionnaire()
        return
    
    risk_score = st.session_state['risk_score']
    risk_level = "Conservative" if risk_score < 4 else \
                 "Moderate" if risk_score < 7 else "Aggressive"
    risk_class = "risk-low" if risk_score < 4 else "risk-medium" if risk_score < 7 else "risk-high"
    
    st.markdown(f"""
    <div class='metric-card'>
        <h2 class='{risk_class}'>Your Risk Score: {risk_score}/10 ({risk_level})</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk profile summary
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
    <div class='metric-card'>
        <h4>Risk Tolerance</h4>
        <p>{risk_level}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
    <div class='metric-card'>
        <h4>Recommended Stocks</h4>
        <p>{"Large-Cap" if risk_score < 4 else "Mix" if risk_score < 7 else "Growth"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
    <div class='metric-card'>
        <h4>Volatility Tolerance</h4>
        <p>{"Low" if risk_score < 4 else "Medium" if risk_score < 7 else "High"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Portfolio recommendation
    if risk_score < 4:
        st.header("🛡️ Conservative Portfolio Recommendation")
        st.markdown("""
        <div class='metric-card'>
            <h4>Asset Allocation</h4>
            <p>- <strong>60% Bonds/Fixed Income</strong>: Government securities, corporate bonds with high ratings</p>
            <p>- <strong>30% Large-Cap Stocks</strong>: Blue-chip companies with stable dividends</p>
            <p>- <strong>10% Cash</strong>: For liquidity and emergency funds</p>
            
            <h4 style='margin-top: 15px;'>Investment Strategy</h4>
            <p>Focus on capital preservation with steady, predictable returns. Avoid high-volatility assets.
            Rebalance portfolio annually to maintain risk profile.</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif risk_score < 7:
        st.header("⚖️ Moderate Portfolio Recommendation")
        st.markdown("""
        <div class='metric-card'>
            <h4>Asset Allocation</h4>
            <p>- <strong>40% Large-Cap Stocks</strong>: Established companies with growth potential</p>
            <p>- <strong>30% Mid-Cap Stocks</strong>: Growing companies with reasonable valuations</p>
            <p>- <strong>20% Bonds</strong>: For stability and income generation</p>
            <p>- <strong>10% International</strong>: Diversification across global markets</p>
            
            <h4 style='margin-top: 15px;'>Investment Strategy</h4>
            <p>Balanced approach between growth and safety. Regular monitoring (quarterly) to adjust 
            allocations based on market conditions. Consider index funds for core holdings.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.header("🚀 Aggressive Portfolio Recommendation")
        st.markdown("""
        <div class='metric-card'>
            <h4>Asset Allocation</h4>
            <p>- <strong>50% Growth Stocks</strong>: High-growth potential companies</p>
            <p>- <strong>20% Small-Cap</strong>: Emerging companies with high growth potential</p>
            <p>- <strong>20% International</strong>: Exposure to global growth opportunities</p>
            <p>- <strong>10% Alternative Investments</strong>: REITs, commodities, or cryptocurrencies</p>
            
            <h4 style='margin-top: 15px;'>Investment Strategy</h4>
            <p>Focus on long-term capital growth. Can tolerate short-term volatility. 
            Active monitoring recommended. Consider sector rotation strategies.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📥 Download Your Report")
    
    if st.button("Generate PDF Report"):
        pdf_data = generate_pdf_report(risk_score)
        st.download_button(
            label="Download Risk Report",
            data=pdf_data,
            file_name="investment_risk_report.pdf",
            mime="application/pdf"
        )

def main():
    st.set_page_config(
        page_title="Investment Risk Analyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'dashboard'
    if 'risk_score' not in st.session_state:
        st.session_state['risk_score'] = 5  # Default medium risk
    
    # Page routing
    if st.session_state['page'] == 'dashboard':
        dashboard()
    elif st.session_state['page'] == 'questionnaire':
        risk_questionnaire()
    elif st.session_state['page'] == 'report':
        report_page()
    elif st.session_state['page'] == 'analysis' and 'selected_event' in st.session_state:
        analyze_event(st.session_state['selected_event'])
    else:
        dashboard()

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'dashboard'
    if 'risk_score' not in st.session_state:
        st.session_state['risk_score'] = 5  # Default medium risk
    
    # Page routing
    if st.session_state['page'] == 'dashboard':
        dashboard()
    elif st.session_state['page'] == 'questionnaire':
        risk_questionnaire()
    elif st.session_state['page'] == 'report':
        report_page()
    elif st.session_state['page'] == 'analysis' and 'selected_event' in st.session_state:
        analyze_event(st.session_state['selected_event'])
    else:
        dashboard()

if __name__ == '__main__':
    main()
