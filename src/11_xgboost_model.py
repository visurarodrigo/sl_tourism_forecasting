import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')

print("="*70)
print("PHASE 6: BUILDING XGBOOST MODEL (MACHINE LEARNING)")
print("="*70)

# 1. Load Data
print("\nLoading data...")
df = pd.read_csv(data_path, parse_dates=['Date'])
print(f"✓ Data loaded: {len(df)} months")

# ==========================================
# 2. FEATURE ENGINEERING (The Secret Sauce)
# ==========================================
print("\nEngineering Time Series Features...")

# Sort by date just in case
df = df.sort_values('Date').reset_index(drop=True)

# A. Lag Features (Past values)
df['Lag_1'] = df['Arrivals'].shift(1)       # Last month
df['Lag_12'] = df['Arrivals'].shift(12)     # Same month last year

# B. Rolling Window Features (Trends and Volatility)
df['Rolling_Mean_3'] = df['Arrivals'].shift(1).rolling(window=3).mean()  # 3-month trend
df['Rolling_Std_3'] = df['Arrivals'].shift(1).rolling(window=3).std()    # 3-month volatility

# C. Time Features (Cyclical patterns)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Drop NaN values created by the shift/rolling functions
# (The first 12-13 rows will have missing data because we can't look back that far)
df_ml = df.dropna().copy()

print(f"✓ Created features. Shape after dropping NaNs: {df_ml.shape}")

# Define Features (X) and Target (y)
features = ['Lag_1', 'Lag_12', 'Rolling_Mean_3', 'Rolling_Std_3', 'Month', 'Year']
X = df_ml[features]
y = df_ml['Arrivals']

# ==========================================
# 3. TRAIN/TEST SPLIT (Chronological)
# ==========================================
# We must split chronologically. Our test set is the last 12 months.
test_size = 12
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# We also need the dates for plotting later
test_dates = df_ml['Date'].iloc[-test_size:]

print(f"✓ Training set: {len(X_train)} months")
print(f"✓ Test set: {len(X_test)} months")

# ==========================================
# 4. BUILD AND TRAIN XGBOOST
# ==========================================
print("\nTraining XGBoost model...")
# XGBoost Regressor for continuous numbers
model = xgb.XGBRegressor(
    n_estimators=100,      # Number of trees
    learning_rate=0.1,     # Step size
    max_depth=3,           # Prevent overfitting
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 5. PREDICT & EVALUATE
# ==========================================
print("\nGenerating forecasts...")
predictions = model.predict(X_test)

actual = y_test.values
rmse = np.sqrt(mean_squared_error(actual, predictions))
mae = mean_absolute_error(actual, predictions)
mape = np.mean(np.abs((actual - predictions) / actual)) * 100

print("\n" + "="*70)
print("XGBOOST MODEL PERFORMANCE METRICS")
print("="*70)
print(f"✓ RMSE: {rmse:,.2f}")
print(f"✓ MAE: {mae:,.2f}")
print(f"✓ MAPE: {mape:.2f}%")

# ==========================================
# 6. FEATURE IMPORTANCE (Crucial for Portfolio!)
# ==========================================
print("\nGenerating Feature Importance Plot...")
importance = model.feature_importances_
feature_names = features

# Create a DataFrame for easy sorting
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_imp_df)))
ax.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1], color=colors)
ax.set_title('XGBoost Feature Importance', fontsize=16, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ==========================================
# 7. VISUALIZE FORECAST vs ACTUAL
# ==========================================
print("\nGenerating Forecast vs Actual Plot...")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the training data context (last 24 months of training)
context_dates = df_ml['Date'].iloc[-(test_size+24):-test_size]
context_actual = y_train.iloc[-24:]
ax.plot(context_dates, context_actual, label='Training Data (Context)', color='royalblue', linewidth=2)

# Plot Test Actual vs Predicted
ax.plot(test_dates, actual, label='Actual (Test)', color='green', linewidth=2, marker='o')
ax.plot(test_dates, predictions, label='XGBoost Forecast', color='red', linewidth=2, marker='x')

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Tourist Arrivals', fontsize=12, fontweight='bold')
ax.set_title('XGBoost: Actual vs Forecast', fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 8. THE FINAL SHOWDOWN
# ==========================================
print("\n" + "="*70)
print("THE FINAL MODEL SHOWDOWN")
print("="*70)
print(f"1. SARIMA  (Statistical) : 13.03% MAPE")
print(f"2. Prophet (Business)    : 17.68% MAPE")
print(f"3. XGBoost (Machine L.)  : {mape:.2f}% MAPE")

# Determine the winner
models = {'SARIMA': 13.03, 'Prophet': 17.68, 'XGBoost': mape}
winner = min(models, key=models.get)
print(f"\n🏆 WINNER: {winner} with {models[winner]:.2f}% MAPE!")
print("="*70)