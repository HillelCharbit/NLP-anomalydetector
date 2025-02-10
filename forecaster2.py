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

# Load the Delta T data
filename_delta_t = "sin_data2.csv"
df_delta_t = pd.read_csv(filename_delta_t)

# Prepare the context for prediction
context_delta_t = torch.tensor(df_delta_t["Difference"])

embedding, tokenizer_state = pipeline.embed(context_delta_t)

prediction_length_delta_t = 5
forecast_delta_t = pipeline.predict(context_delta_t, prediction_length_delta_t)  # shape [num_series, num_samples, prediction_length]

# Analyze Delta T forecast
low_delta_t, median_delta_t, high_delta_t = np.quantile(forecast_delta_t[0].numpy(), [0.1, 0.5, 0.9], axis=0)
print("Delta T Forecast Median:", median_delta_t)

# Use continuous median values instead of rounding
median_rounded = median_delta_t  # Avoid cumulative rounding errors
print("Continuous Median Delta T:", median_rounded)

# Load the sine wave data
filename = "sin_data.xlsx"
df = pd.read_excel(filename)

# Add periodicity features to the sine wave data
df["Sin_Index"] = np.sin(df["X"])
context = torch.tensor(df["Sin(X)"])

# Calculate dynamic prediction length
mean_delta_t = np.mean(median_delta_t)
prediction_length = int(mean_delta_t * prediction_length_delta_t) + 1
print("HEre")

forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# Analyze sine wave forecast
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
# Predict anomaly locations using continuous Delta T
x_column = [df_delta_t["Index"].iloc[-1]]  # Start from the last index value in Delta T
y_column = [df["Sin(X)"].iloc[x_column[0]]]  # Corresponding sine value
y_median_index = x_column[0] - len(df)

for i in median_rounded:
    y_median_index += i  # Use continuous values
    x_column.append(x_column[-1] + i)
    y_column.append(median[int(y_median_index)])

# Plot historical and forecasted data
plt.figure(figsize=(10, 6))

# Plot last 50 historical points for sine wave
last_50_index = range(len(df) - 50, len(df))
last_50_data = df["Sin(X)"].iloc[-50:]
plt.plot(last_50_index, last_50_data, color="royalblue", label="Last 50 historical data")

# Plot sine wave forecast
forecast_index = range(len(df), len(df) + prediction_length)
plt.plot(forecast_index, median, color="tomato", label="Median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")

# Overlay predicted anomaly locations
plt.scatter(x_column, y_column, color="green", label="Forecasted anomalies")

# Add true sine function for comparison
true_sine = np.sin(np.linspace(0, 2 * np.pi * (len(df) + prediction_length) / len(df), len(df) + prediction_length))
plt.plot(range(len(df) + prediction_length), true_sine, label="True sine function", linestyle="dotted", color="black")

# Add labels, grid, legend, and title
plt.legend()
plt.grid()
plt.xlabel("Index")
plt.ylabel("Sin(X)")
plt.title("Sine Wave Forecast with Anomaly Detection")
plt.show()