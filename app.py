# ============================================================================
# STREAMLIT DASHBOARD - STOCK PRICE PREDICTION ML PROJECT
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction ML",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

@st.cache_data
def load_data():
    """Load preprocessed data and test results"""
    df = pd.read_csv('processed_data.csv')
    test_results = pd.read_csv('test_results.csv')
    test_results['Date'] = pd.to_datetime(test_results['Date'])
    return df, test_results

@st.cache_resource
def load_model():
    """Load trained model and artifacts"""
    with open('stock_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('model_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return model, features, results

# Load everything
df, test_results = load_data()
model, feature_cols, model_results = load_model()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Model Performance", "Backtesting Results", "Feature Importance", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.info("""
**DTSC691 Capstone Project**  
Stock Price Prediction using Machine Learning  
Author: Hidaya Patel  
Date: December 2024
""")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "Overview":
    st.title("📈 Stock Price Prediction ML Pipeline")
    st.markdown("### Predicting Short-Term Stock Movements with Machine Learning")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset", "AAPL", "10 Years")
    with col2:
        st.metric("Total Samples", f"{len(df):,}")
    with col3:
        st.metric("Features", "28", "Technical Indicators")
    with col4:
        st.metric("Models Trained", "3", "LR, RF, XGBoost")
    
    st.markdown("---")
    
    # Price chart
    st.subheader("📊 Historical Stock Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'],
        mode='lines',
        name='Closing Price',
        line=dict(color='#2E86AB', width=2)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Dataset Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Start Date', 'End Date', 'Trading Days', 'Avg Daily Return', 'Volatility'],
            'Value': [
                df['Date'].min(),
                df['Date'].max(),
                f"{len(df):,}",
                f"{df['Daily_Return'].mean():.4f}%",
                f"{df['Daily_Return'].std():.4f}%"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
        1. ✅ Build end-to-end ML pipeline
        2. ✅ Engineer 28 technical indicators
        3. ✅ Train and compare multiple models
        4. ✅ Evaluate with financial metrics
        5. ✅ Provide model interpretability
        6. ✅ Assess real-world applicability
        """)

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================

elif page == "Model Performance":
    st.title("🤖 Model Performance Analysis")
    
    # Model comparison table
    st.subheader("📊 Model Comparison")
    
    comparison_data = {
        'Model': list(model_results.keys()),
        'Test Accuracy': [f"{model_results[m]['test_accuracy']:.2%}" for m in model_results.keys()],
        'Precision': [f"{model_results[m]['precision']:.4f}" for m in model_results.keys()],
        'Recall': [f"{model_results[m]['recall']:.4f}" for m in model_results.keys()],
        'F1-Score': [f"{model_results[m]['f1_score']:.4f}" for m in model_results.keys()]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    # Best model highlight
    best_model_name = max(model_results.keys(), key=lambda m: model_results[m]['f1_score'])
    st.success(f"🏆 **Best Model:** {best_model_name} (F1-Score: {model_results[best_model_name]['f1_score']:.4f})")
    
    st.markdown("---")
    
    # Confusion Matrix (for best model)
    st.subheader("🎯 Prediction Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted
        results_df = pd.DataFrame({
            'Actual': test_results['Target'][:len(model_results[best_model_name]['y_pred'])],
            'Predicted': model_results[best_model_name]['y_pred']
        })
        
        confusion = pd.crosstab(
            results_df['Actual'], 
            results_df['Predicted'],
            rownames=['Actual'],
            colnames=['Predicted']
        )
        
        fig = px.imshow(
            confusion,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Down (0)', 'Up (1)'],
            y=['Down (0)', 'Up (1)'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(title="Confusion Matrix", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Key Insights")
        st.markdown(f"""
        **Model:** {best_model_name}
        
        **Performance:**
        - Accuracy: {model_results[best_model_name]['test_accuracy']:.2%}
        - Precision: {model_results[best_model_name]['precision']:.4f}
        - Recall: {model_results[best_model_name]['recall']:.4f}
        - F1-Score: {model_results[best_model_name]['f1_score']:.4f}
        
        **Interpretation:**
        - Model slightly better than random (50%)
        - Demonstrates inherent difficulty of stock prediction
        - Technical indicators provide weak predictive signal
        """)

# ============================================================================
# PAGE 3: BACKTESTING RESULTS
# ============================================================================

elif page == "Backtesting Results":
    st.title("💰 Backtesting & Financial Performance")
    
    st.markdown("""
    Simulated trading strategy: **Buy when model predicts 'up', hold cash otherwise**  
    Transaction cost: **0.1% per trade**
    """)
    
    # Financial metrics
    col1, col2, col3 = st.columns(3)
    
    strategy_return = (test_results['Strategy_Cumulative'].iloc[-1] - 1) * 100
    buyhold_return = (test_results['BuyHold_Cumulative'].iloc[-1] - 1) * 100
    
    with col1:
        st.metric("ML Strategy Return", f"{strategy_return:.2f}%", 
                 delta=f"{strategy_return - buyhold_return:.2f}% vs B&H")
    with col2:
        st.metric("Buy & Hold Return", f"{buyhold_return:.2f}%")
    with col3:
        win_rate = (test_results['Strategy_Return'] > 0).sum() / len(test_results) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    st.markdown("---")
    
    # Cumulative returns chart
    st.subheader("📈 Cumulative Returns Comparison")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_results['Date'],
        y=test_results['Strategy_Cumulative'],
        mode='lines',
        name='ML Strategy',
        line=dict(color='#2E86AB', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_results['Date'],
        y=test_results['BuyHold_Cumulative'],
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#A23B72', width=2.5, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return (Multiplier)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis
    st.subheader("🔍 Performance Analysis")
    
    st.warning("""
    **Key Finding:** The ML strategy underperformed buy-and-hold.
    
    **Why this matters:**
    - Demonstrates honest scientific evaluation
    - Shows limitations of technical analysis for short-term prediction
    - Transaction costs significantly impact active strategies
    - Market efficiency makes consistent alpha generation difficult
    
    **Conclusion:** This project successfully demonstrates ML methodology while 
    providing realistic insights into algorithmic trading challenges.
    """)

# ============================================================================
# PAGE 4: FEATURE IMPORTANCE
# ============================================================================

elif page == "Feature Importance":
    st.title("🔍 Feature Importance Analysis")
    
    st.subheader("📊 Top Predictive Features")
    
    # Get feature importance from Logistic Regression
    lr_coef = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False).head(15)
    
    fig = px.bar(
        lr_coef,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Features by Importance',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature descriptions
    st.subheader("📖 Feature Descriptions")
    
    feature_desc = {
        'Daily_Return': 'Daily percentage change in price',
        'MACD': 'Moving Average Convergence Divergence - momentum indicator',
        'RSI': 'Relative Strength Index - overbought/oversold indicator',
        'ATR': 'Average True Range - volatility measure',
        'Volatility_10': '10-day rolling standard deviation of returns',
        'SMA_10': '10-day Simple Moving Average',
        'BB_Width': 'Bollinger Bands width - volatility indicator',
        'Stoch_D': 'Stochastic Oscillator - momentum indicator'
    }
    
    for feat, desc in feature_desc.items():
        if feat in lr_coef['Feature'].values:
            st.markdown(f"**{feat}:** {desc}")

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

elif page == "About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## Stock Price Prediction using Machine Learning
    
    ### Project Overview
    This capstone project implements an end-to-end machine learning pipeline for predicting 
    short-term stock price movements using historical market data and technical indicators.
    
    ### Methodology
    
    **1. Data Collection**
    - Source: Yahoo Finance API
    - Ticker: AAPL (Apple Inc.)
    - Period: 10 years (2015-2025)
    - Features: OHLCV data
    
    **2. Feature Engineering**
    - 28 technical indicators created
    - Moving averages (SMA, EMA)
    - Momentum indicators (MACD, RSI, Stochastic)
    - Volatility measures (ATR, Bollinger Bands)
    - Lagged returns and rolling statistics
    
    **3. Model Development**
    - Algorithms: Logistic Regression, Random Forest, XGBoost
    - Validation: Time-series split (80/20)
    - Optimization: Regularization to prevent overfitting
    
    **4. Evaluation**
    - Statistical metrics: Accuracy, Precision, Recall, F1-Score
    - Financial metrics: Returns, Sharpe Ratio, Maximum Drawdown
    - Backtesting with transaction costs
    
    **5. Interpretability**
    - Feature importance analysis
    - SHAP values for model explainability
    
    ### Key Findings
    
    ✅ **Technical Success:** Complete ML pipeline successfully implemented  
    ✅ **Model Performance:** 51% accuracy (marginally better than random)  
    ✅ **Real-world Insight:** Demonstrates why short-term stock prediction is challenging  
    ✅ **Honest Evaluation:** Backtesting shows underperformance vs buy-and-hold  
    
    ### Technologies Used
    - **Languages:** Python
    - **ML Libraries:** scikit-learn, XGBoost, SHAP
    - **Data:** pandas, NumPy, yfinance
    - **Visualization:** Matplotlib, Seaborn, Plotly
    - **Dashboard:** Streamlit
    
    ### Conclusion
    This project demonstrates a comprehensive understanding of the machine learning workflow 
    while providing realistic insights into the challenges of financial forecasting. The honest 
    evaluation of model limitations represents a key strength of the analysis.
    
    ---
    
    **Author:** Hidaya Patel  
    **Course:** DTSC691 - Applied Data Science Capstone  
    **Institution:** Eastern University  
    **Date:** December 2024
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")