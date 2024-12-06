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

# Load the sine data
filename = "sin_data.xlsx"
df = pd.read_excel(filename, engine="openpyxl")

# Prepare the context for prediction
context = torch.tensor(df["Sin(X)"])
prediction_length = 12
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# Slice the last 50 historical points
last_50_index = range(len(df) - 50, len(df))
last_50_data = df["Sin(X)"].iloc[-50:]

# Forecast data
forecast_index = range(len(df), len(df) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# Plot the results
plt.figure(figsize=(8, 4))
plt.plot(last_50_index, last_50_data, color="royalblue", label="Last 50 historical data")
plt.plot(forecast_index, median, color="tomato", label="Median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.xlabel("Index")
plt.ylabel("Sin(X)")
plt.title("Last 50 Points of Historical Data with Forecast")
plt.show()
