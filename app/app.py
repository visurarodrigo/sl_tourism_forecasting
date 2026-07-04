import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Config ---
st.set_page_config(page_title="SL Tourism Forecasting", page_icon="🇱🇰", layout="wide")

# --- Path Setup (Bulletproof) ---
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)  # Go UP one level from 'app/' to project root

data_dir = os.path.join(project_root, 'Data', '03_model_outputs')
country_data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_country_wise_final.csv')
clean_data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')

# --- Load Data ---
@st.cache_data
def load_data():
    historical = pd.read_csv(clean_data_path, parse_dates=['Date'])
    historical = historical.rename(columns={'Date': 'ds', 'Arrivals': 'y'})
    
    sarima = pd.read_csv(os.path.join(data_dir, 'forecast_sarima.csv'), parse_dates=['ds'])
    prophet = pd.read_csv(os.path.join(data_dir, 'forecast_prophet.csv'), parse_dates=['ds'])
    xgboost = pd.read_csv(os.path.join(data_dir, 'forecast_xgboost.csv'), parse_dates=['ds'])
    
    metrics = pd.read_csv(os.path.join(data_dir, 'model_metrics.csv'))
    feat_imp = pd.read_csv(os.path.join(data_dir, 'xgboost_feature_importance.csv'))
    seasonality = pd.read_csv(os.path.join(data_dir, 'prophet_seasonality.csv'), parse_dates=['ds'])
    
    countries = pd.read_csv(country_data_path, parse_dates=['Date'])
    
    return historical, sarima, prophet, xgboost, metrics, feat_imp, seasonality, countries

historical, sarima, prophet, xgboost, metrics, feat_imp, seasonality, countries = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🇱🇰 SL Tourism Intelligence")
page = st.sidebar.radio("Navigate", ["Executive Overview", "Forecast Explorer", "Model Diagnostics", "Source Markets"])
# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "Executive Overview":
    st.header("Executive Overview")
    
    best_model = metrics.loc[metrics['MAPE'].idxmin()]
    consensus_12m = (sarima['yhat'].sum() + prophet['yhat'].sum() + xgboost['yhat'].sum()) / 3
    peak_month = sarima.loc[sarima['yhat'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Performing Model", best_model['Model'], f"{best_model['MAPE']:.2f}% MAPE")
    col2.metric("Consensus Forecast (Next 24 Mo.)", f"{consensus_12m:,.0f}", "Avg. of 3 Models")
    col3.metric("Forecasted Peak Month", peak_month['ds'].strftime('%b %Y'), f"{peak_month['yhat']:,.0f} Arrivals")
    
    st.markdown("---")
    st.subheader("Historical Trend & Consensus Forecast")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'], mode='lines', name='Historical', line=dict(color='#1f77b4', width=3)))
    
    # Add average forecast
    avg_forecast = (sarima['yhat'].values + prophet['yhat'].values + xgboost['yhat'].values) / 3
    fig.add_trace(go.Scatter(x=sarima['ds'], y=avg_forecast, mode='lines', name='Consensus Forecast', 
                             line=dict(color='#ff7f0e', width=3, dash='dash')))
    
    fig.update_layout(template="plotly_white", height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: FORECAST EXPLORER
# ==========================================
elif page == "Forecast Explorer":
    st.header("Interactive Forecast Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Model", ["Prophet", "SARIMA", "XGBoost"])
    with col2:
        ci_choice = st.selectbox("Confidence Interval", ["95%", "80%"])
        
    if model_choice == "SARIMA": forecast_df = sarima
    elif model_choice == "Prophet": forecast_df = prophet
    else: forecast_df = xgboost
        
    # Dynamic CI adjustment
    mean = forecast_df['yhat']
    upper_95 = forecast_df['yhat_upper']
    std_dev = (upper_95 - mean) / 1.96
    
    if ci_choice == "80%":
        lower_col = mean - 1.28 * std_dev
        upper_col = mean + 1.28 * std_dev
    else:
        lower_col = forecast_df['yhat_lower']
        upper_col = forecast_df['yhat_upper']

    # Combine for continuous line
    hist_plot = historical[['ds', 'y']].copy()
    fore_plot = forecast_df[['ds', 'yhat']].copy().rename(columns={'yhat': 'y'})
    last_hist = hist_plot.iloc[[-1]].copy()
    fore_plot = pd.concat([last_hist, fore_plot], ignore_index=True)
    
    fig = go.Figure()
    
    # Confidence Ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
        y=pd.concat([upper_col, lower_col[::-1]]),
        fill='toself', fillcolor='rgba(68, 136, 204, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), name=f'{ci_choice} CI'
    ))
    
    fig.add_trace(go.Scatter(x=hist_plot['ds'], y=hist_plot['y'], mode='lines', name='Historical', line=dict(color='#1f77b4', width=3)))
    fig.add_trace(go.Scatter(x=fore_plot['ds'], y=fore_plot['y'], mode='lines', name=f'{model_choice} Forecast', line=dict(color='#ff7f0e', width=3, dash='dash')))
    
    fig.update_layout(
        template="plotly_white", height=600, hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 3: MODEL DIAGNOSTICS
# ==========================================
elif page == "Model Diagnostics":
    st.header("Model Diagnostics & Performance")
    
    st.subheader("Model Comparison Metrics")
    st.dataframe(metrics.style.format({"RMSE": "{:,.2f}", "MAE": "{:,.2f}", "MAPE": "{:.2f}%"}), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("XGBoost Feature Importance")
        fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                         title="What drives the ML model?", color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col2:
        st.subheader("Prophet Yearly Seasonality")
        fig_season = px.line(seasonality, x='ds', y='yearly', title="Yearly Seasonal Pattern")
        st.plotly_chart(fig_season, use_container_width=True)

# ==========================================
# PAGE 4: SOURCE MARKETS
# ==========================================
elif page == "Source Markets":
    st.header("Source Market Insights")
    
    top_10 = countries.groupby('Country')['Arrivals'].sum().nlargest(10).reset_index()
    fig_top = px.bar(top_10, x='Arrivals', y='Country', orientation='h', 
                     title="Top 10 Source Countries (All Time)", color='Arrivals', color_continuous_scale='Viridis')
    st.plotly_chart(fig_top, use_container_width=True)
    
    top_5_countries = top_10['Country'].head(5).tolist()
    df_trends = countries[countries['Country'].isin(top_5_countries)]
    
    fig_trends = px.line(df_trends, x='Date', y='Arrivals', color='Country',
                         title="Recovery Trends: Top 5 Source Countries", markers=True)
    st.plotly_chart(fig_trends, use_container_width=True)