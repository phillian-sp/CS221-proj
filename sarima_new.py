import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import pmdarima as pm

warnings.filterwarnings("ignore")

# 1. Load & aggregate
CSV_PATH = "data/all_usage_halfhour.csv"
raw = pd.read_csv(CSV_PATH, parse_dates=["time"])

# Sum every numeric company column for each 30-minute stamp
total_usage = raw.drop(columns=["time"]).sum(axis=1, skipna=True)

# Build half-hourly series with frequency
y = pd.Series(total_usage.values, index=raw["time"]).asfreq("30min").dropna()
# Remove last three rows (NaN values)
y = y[:-3]  # remove last two rows (NaN values)

# Load data (assuming y is already defined from previous cells)
# Split into initial training and test sets
test_size = 48  # 24 hours of 30-min data
train = y[:-test_size]
test = y[-test_size:]

# 1. search the hyper-parameter space on the *training* set
auto_res = pm.auto_arima(
    train,
    seasonal=True, m=48,          # 48 half-hours = 1 day
    d=0, D=1,
    start_p=0, start_q=0, max_p=2, max_q=2,
    start_P=0, start_Q=0, max_P=2, max_Q=2,
    information_criterion="aic",
    stepwise=True, trace=True,    # prints progress
    error_action="ignore",        # skip models that fail to converge
    suppress_warnings=True
)
print(auto_res.summary())         # look at chosen orders

# 2. extract the tuples
best_order         = auto_res.order            # (p,d,q)
best_seasonal      = auto_res.seasonal_order   # (P,D,Q,s)

# 3. refit *exactly* that model in statsmodels (gives full state-space access)
sarimax_model = SARIMAX(
    train,
    order=best_order,
    seasonal_order=best_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False,
    measurement_error=True,
    trend='c'
).fit(disp=False)

# sarimax_model = SARIMAX(
#     train,
#     order=(2, 0, 1),
#     seasonal_order=(2, 1, 1, 48),
#     enforce_stationarity=False,
#     enforce_invertibility=False,
#     measurement_error=True,
#     trend='c'
# ).fit(disp=False)

# 4. Kalman-filter rolling loop
predictions, actuals = [], []
for t in range(len(test)):
    actual = test.iloc[t]
    actuals.append(actual)

    forecast = sarimax_model.forecast(steps=1)[0]
    predictions.append(forecast)

    if t < len(test) - 1:                      # update state with new obs
        sarimax_model = sarimax_model.append(pd.Series([actual], index=[test.index[t]]))

# Convert to Series for easier plotting
predictions = pd.Series(predictions, index=test.index)
actuals = pd.Series(actuals, index=test.index)

# Calculate error metrics
errors = predictions - actuals
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
mape = np.mean(np.abs(errors / actuals)) * 100

print("\nRolling one-step forecast metrics (using Kalman filter):")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.2f}%")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(actuals.index, actuals.values, label='Actual', linewidth=2)
plt.plot(predictions.index, predictions.values, '--', label='Rolling Forecast', linewidth=2)
plt.title('24-hour Rolling Forecast using Kalman Filter (No Refitting)')
plt.xlabel('Time')
plt.ylabel('Compressed Air Usage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()