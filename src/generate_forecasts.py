import pandas as pd
import numpy as np
import os
import warnings
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# --- 1. Setup Paths (Bulletproof) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # This goes UP one level from 'src/' to the main project folder

data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')
output_dir = os.path.join(project_root, 'Data', '03_model_outputs')
os.makedirs(output_dir, exist_ok=True)

# The rest of your code remains exactly the same...
df = pd.read_csv(data_path, parse_dates=['Date'])

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

test_size = 12
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]
metrics = []

# --- 2. SARIMA ---
print("Training SARIMA...")
model_sarima = SARIMAX(train['Arrivals'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False, enforce_invertibility=False)
results_sarima = model_sarima.fit(disp=False)

# Forecast test set for metrics
pred_sarima_test = results_sarima.get_forecast(steps=test_size)
sarima_test_pred = pred_sarima_test.predicted_mean.values

# Forecast future 24 months
future_steps = 24
pred_sarima_future = results_sarima.get_forecast(steps=future_steps)
sarima_future_dates = pd.date_range(start=test['Date'].max() + pd.DateOffset(months=1), periods=future_steps, freq='MS')

sarima_future_df = pd.DataFrame({
    'ds': sarima_future_dates,
    'yhat': pred_sarima_future.predicted_mean.values,
    'yhat_lower': pred_sarima_future.conf_int(alpha=0.05).iloc[:, 0].values,
    'yhat_upper': pred_sarima_future.conf_int(alpha=0.05).iloc[:, 1].values,
    'model': 'SARIMA'
})
sarima_future_df.to_csv(os.path.join(output_dir, 'forecast_sarima.csv'), index=False)

rmse = np.sqrt(mean_squared_error(test['Arrivals'], sarima_test_pred))
mae = mean_absolute_error(test['Arrivals'], sarima_test_pred)
mape = np.mean(np.abs((test['Arrivals'] - sarima_test_pred) / test['Arrivals'])) * 100
metrics.append({'Model': 'SARIMA', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

# --- 3. PROPHET ---
print("Training Prophet...")
df_prophet = df.rename(columns={'Date': 'ds', 'Arrivals': 'y'})
train_p = df_prophet.iloc[:-test_size]
test_p = df_prophet.iloc[-test_size:]

model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.1)
model_prophet.fit(train_p)

# Forecast test set for metrics
future_test = model_prophet.make_future_dataframe(periods=test_size, freq='MS')
forecast_test = model_prophet.predict(future_test)
prophet_test_pred = forecast_test[forecast_test['ds'].isin(test_p['ds'])]['yhat'].values

# Forecast future 24 months
future_future = model_prophet.make_future_dataframe(periods=future_steps, freq='MS')
forecast_future = model_prophet.predict(future_future)
prophet_future_df = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_steps).copy()
prophet_future_df['model'] = 'Prophet'
prophet_future_df.to_csv(os.path.join(output_dir, 'forecast_prophet.csv'), index=False)

# Save yearly seasonality for diagnostics
yearly_comp = forecast_future[['ds', 'yearly']].copy()
yearly_comp.to_csv(os.path.join(output_dir, 'prophet_seasonality.csv'), index=False)

rmse = np.sqrt(mean_squared_error(test_p['y'], prophet_test_pred))
mae = mean_absolute_error(test_p['y'], prophet_test_pred)
mape = np.mean(np.abs((test_p['y'] - prophet_test_pred) / test_p['y'])) * 100
metrics.append({'Model': 'Prophet', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

# --- 4. XGBOOST ---
print("Training XGBoost...")
df_ml = df.copy()
df_ml['Lag_1'] = df_ml['Arrivals'].shift(1)
df_ml['Lag_12'] = df_ml['Arrivals'].shift(12)
df_ml['Rolling_Mean_3'] = df_ml['Arrivals'].shift(1).rolling(window=3).mean()
df_ml['Rolling_Std_3'] = df_ml['Arrivals'].shift(1).rolling(window=3).std()
df_ml['Month'] = df_ml['Date'].dt.month
df_ml['Year'] = df_ml['Date'].dt.year
df_ml = df_ml.dropna().reset_index(drop=True)

features = ['Lag_1', 'Lag_12', 'Rolling_Mean_3', 'Rolling_Std_3', 'Month', 'Year']
X = df_ml[features]
y = df_ml['Arrivals']

X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_xgb.fit(X_train, y_train)
xgb_test_pred = model_xgb.predict(X_test)

# Iterative future forecasting for XGBoost
last_known_date = df['Date'].max()
future_dates = pd.date_range(start=last_known_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
history = df['Arrivals'].tolist()
xgb_preds = []

for i in range(future_steps):
    lag_1 = history[-1]
    lag_12 = history[-12]
    roll_mean_3 = np.mean(history[-3:])
    roll_std_3 = np.std(history[-3:], ddof=1)
    month = future_dates[i].month
    year = future_dates[i].year
    
    X_future = pd.DataFrame([[lag_1, lag_12, roll_mean_3, roll_std_3, month, year]], columns=features)
    pred = model_xgb.predict(X_future)[0]
    xgb_preds.append(pred)
    history.append(pred)

# Estimate Confidence Intervals using test residuals
residuals = y_test.values - xgb_test_pred
std_res = np.std(residuals)

xgb_future_df = pd.DataFrame({
    'ds': future_dates,
    'yhat': xgb_preds,
    'yhat_lower': np.array(xgb_preds) - 1.96 * std_res,
    'yhat_upper': np.array(xgb_preds) + 1.96 * std_res,
    'model': 'XGBoost'
})
xgb_future_df.to_csv(os.path.join(output_dir, 'forecast_xgboost.csv'), index=False)

# Feature Importance
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': model_xgb.feature_importances_
}).sort_values(by='Importance', ascending=False)
feat_imp.to_csv(os.path.join(output_dir, 'xgboost_feature_importance.csv'), index=False)

rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
mae = mean_absolute_error(y_test, xgb_test_pred)
mape = np.mean(np.abs((y_test - xgb_test_pred) / y_test)) * 100
metrics.append({'Model': 'XGBoost', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

# Save Metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)

print("✅ All forecasts, metrics, and diagnostics saved to Data/03_model_outputs/!")