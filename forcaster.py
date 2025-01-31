import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

# Load the Chronos pipeline
pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-tiny",
  device_map="cpu",
  torch_dtype=torch.bfloat16,
)

# Load the delta t data
filename_delta_t = "sin_data2.csv"
df_delta_t = pd.read_csv(filename_delta_t)
df_delta_t = df_delta_t.dropna().reset_index(drop=True)
print(df_delta_t.head())

# Prepare the context for prediction
context_delta_t = torch.tensor(df_delta_t["Difference"])
prediction_length_delta_t = 5
forecast_delta_t = pipeline.predict(context_delta_t, prediction_length_delta_t)  # shape [num_series, num_samples, prediction_length]

print(forecast_delta_t[0].numpy())

# Slice the last 50 historical points
last_50_index_delta_t = range(len(df_delta_t) - 50, len(df_delta_t))
last_50_data_delta_t = df_delta_t["Difference"].iloc[-50:]

# Forecast data
forecast_index_delta_t = range(len(df_delta_t), len(df_delta_t) + prediction_length_delta_t)
_, median, _ = np.quantile(forecast_delta_t[0].numpy(), [0.1, 0.5, 0.9], axis=0)

median_rounded = np.round(median)
print(median)

# Load the sin data 
filename = "sin_data.xlsx"
df = pd.read_excel(filename)

# Prepare the context for prediction
context = torch.tensor(df["Sin(X)"])
max_median_rounded = int(max(median_rounded))
print(f"Max Median Rounded: {max_median_rounded}")
prediction_length = 5 *  max_median_rounded + 1
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# Slice the last 50 historical points
last_50_index = range(len(df) - 50, len(df))
last_50_data = df["Sin(X)"].iloc[-50:]

# Forecast data
forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)


x_column = [df_delta_t["Index"].iloc[-1]]  # Initialize with the last index value
y_column = [df["Sin(X)"].iloc[x_column[0]]]  # Initialize with the last difference value
y_median_index = x_column[0] - len(df)

for i in median_rounded:
    y_median_index += int(i)
    x_column.append(x_column[-1] + int(i))
    y_column.append(median[y_median_index])


# Plot the results
plt.figure(figsize=(8, 4))
plt.plot(last_50_index, last_50_data, color="royalblue", label="Last 50 historical data")
plt.plot(forecast_index, median, color="tomato", label="Median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")

plt.scatter(x_column, y_column, color="green", label="Forecast")


plt.legend()
plt.grid()
plt.xlabel("Index")
plt.ylabel("Difference")
plt.title("Last 50 Points of Historical Data with Forecast")
plt.show()
