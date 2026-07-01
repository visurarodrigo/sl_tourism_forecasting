import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings

# Suppress Prophet's verbose warnings for a cleaner terminal
warnings.filterwarnings('ignore')

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')

print("="*70)
print("PHASE 5: BUILDING PROPHET MODEL")
print("="*70)

# 1. Load and Prepare Data
print("\nLoading data...")
df = pd.read_csv(data_path, parse_dates=['Date'])

# Prophet requires columns to be exactly named 'ds' and 'y'
df_prophet = df.rename(columns={'Date': 'ds', 'Arrivals': 'y'})
print(f"✓ Data loaded and formatted for Prophet: {len(df_prophet)} months")

# 2. Train/Test Split (Same chronological split as SARIMA for fair comparison)
test_size = 12
train = df_prophet.iloc[:-test_size]
test = df_prophet.iloc[-test_size:]

print(f"✓ Training set: {len(train)} months")
print(f"✓ Test set: {len(test)} months")

# 3. Initialize and Fit Prophet
print("\nInitializing Prophet model...")
# We disable weekly/daily seasonality because our data is strictly monthly
model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=False, 
    daily_seasonality=False,
    changepoint_prior_scale=0.1 # Controls how flexibly the trend can adapt to shocks
)

print("Fitting model (this might take 10-20 seconds)...")
model.fit(train)

# 4. Generate Forecasts
print("Generating forecasts...")
# Create a dataframe with dates for the test period
future = model.make_future_dataframe(periods=test_size, freq='MS') # MS = Month Start frequency
forecast = model.predict(future)

# Extract only the predictions for our test period
test_forecast = forecast[forecast['ds'].isin(test['ds'])]

# 5. Calculate Metrics
actual = test['y'].values
predicted = test_forecast['yhat'].values

rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print("\n" + "="*70)
print("PROPHET MODEL PERFORMANCE METRICS")
print("="*70)
print(f"✓ RMSE: {rmse:,.2f}")
print(f"✓ MAE: {mae:,.2f}")
print(f"✓ MAPE: {mape:.2f}%")

# 6. Visualizations
print("\nGenerating Prophet Forecast Plot...")
fig = model.plot(forecast)
plt.title('Prophet Forecast: Actual vs Predicted', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Tourist Arrivals', fontsize=12)

# Highlight the test period in red
plt.axvspan(test['ds'].min(), test['ds'].max(), color='red', alpha=0.15, label='Test Period (Forecast)')
plt.legend()
plt.show()

print("Generating Component Plots (Trend & Seasonality)...")
fig2 = model.plot_components(forecast)
plt.show()

print("\n" + "="*70)
print("PROPHET MODEL COMPLETE!")
print("="*70)
print(f"\n--- MODEL SHOWDOWN SO FAR ---")
print(f"SARIMA MAPE:  13.03%")
print(f"Prophet MAPE: {mape:.2f}%")
print(f"\nNext Step: Build the XGBoost Machine Learning baseline!")