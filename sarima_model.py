"""
half_hour_sarima_forecast.py
============================
Day-ahead (48-step) SARIMA forecast for plant-wide gas demand.
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

###############################################################################
# 1. Load & aggregate
###############################################################################
CSV_PATH = "data/all_usage_halfhour.csv"
raw = pd.read_csv(CSV_PATH, parse_dates=["time"])

# Sum every numeric company column for each 30-minute stamp
total_usage = (
    raw.drop(columns=["time"])           # keep only usage columns
        .sum(axis=1, skipna=True)        # NaNs in any firm are ignored
)

# Build a half-hourly time-series with an explicit frequency
y = pd.Series(total_usage.values, index=raw["time"]).asfreq("30min")

###############################################################################
# 2. Fit SARIMA(p,d,q)(P,D,Q,s)
#    - (1,1,1)(1,1,1,48) is a robust first guess for 30-min data with a
#      strong daily pattern (48 half-hour bins per day).
###############################################################################
model = SARIMAX(
    y,
    order=(1, 1, 1),           # p, d, q
    seasonal_order=(1, 1, 1, 48),  # P, D, Q, s
    trend="c",          
    enforce_stationarity=False,
    enforce_invertibility=False,
    simple_differencing=False,
    measurement_error=True      # handles small NaN gaps gracefully
)
res = model.fit(disp=False)

print("parameters:")
print(res.params)
print("state means:")
print(res.filtered_state.mean(axis=1)[-1])
###############################################################################
# 3. Forecast the next 24 h  (48 × 30-min steps)
###############################################################################
forecast = res.forecast(48)

print("\n================== 24-hour SARIMA forecast ==================")
print(forecast)

# make a plot of the last 48 hours of data and the forecast
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plt.plot(y.index[-48:], y.values[-48:], label="Last 24 hours of data", color="blue")
plt.plot(forecast.index, forecast.values, label="24-hour forecast", color="orange")
plt.axvline(x=y.index[-1], color="gray", linestyle="--", label="Forecast start")
plt.title("24-hour SARIMA Forecast")
plt.xlabel("Time")
plt.ylabel("Pressed Air Demand")
plt.legend()
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M")) 
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gcf().autofmt_xdate()
plt.grid()
plt.show()
# Save the forecast to a CSV file
forecast.to_csv("data/sarima_forecast.csv", index=True, header=["Forecast"])
