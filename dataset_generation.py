import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def generate_dataset(ticker, start_date, end_date):
    data = get_data(ticker, start_date, end_date)
    data['Close_diff'] = data['Close'].diff()
    data['Close_diff'] = data['Close_diff'].shift(-1)
    data['Close_diff'] = data['Close_diff'].fillna(0)
    data['Target'] = np.where(data['Close_diff'] > 0, 1, 0)
    data.dropna(inplace=True)
    return data

def plot_data(data):
    """
    Plots the Close price, MA12, MA24, and highlights the Golden Cross events.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot the Close price and moving averages
    plt.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.7)
    plt.plot(data.index, data['MA12'], label='MA12', color='blue', alpha=0.8)
    plt.plot(data.index, data['MA24'], label='MA24', color='orange', alpha=0.8)
    
    # Identify and mark the golden cross events with green upward arrows.
    golden_cross_points = data[data['GoldenCross'] == 1]
    plt.scatter(golden_cross_points.index, golden_cross_points['Close'],
                label='Golden Cross', marker='^', color='green', s=100)
    
    plt.title('KO Close Price with MA12, MA24, and Golden Cross')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    ticker = 'KO'
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    data = generate_dataset(ticker, start_date, end_date)
    
    # Keep only the relevant column: Close price
    data = data[['Close']]
    print("Head of the Close price data:")
    print(data.head())
    
    # Add moving average columns
    data['MA12'] = data['Close'].rolling(window=12).mean()
    data['MA24'] = data['Close'].rolling(window=24).mean()
    
    # Create the Golden Cross indicator.
    # The indicator is 1 when:
    #   - Today's MA12 is greater than MA24, AND
    #   - Yesterday's MA12 was less than or equal to yesterday's MA24.
    data['GoldenCross'] = np.where(
        (data['MA12'] > data['MA24']) & (data['MA12'].shift(1) <= data['MA24'].shift(1)),
        1,
        0
    )
    
    # Display the last 10 rows of relevant columns
    print("\nLast 10 rows with MA12, MA24, and GoldenCross indicator:")
    print(data[['Close', 'MA12', 'MA24', 'GoldenCross']].tail(10))
    
    # Print the total count of Golden Cross occurrences
    print("\nGolden Cross count:")
    print(data['GoldenCross'].sum())
    
    print("\nData shape:", data.shape)
    
    # Plot the results
    plot_data(data)
    
    # save the data to a CSV file, only the date, close and indicator columns
    data[['Close', 'GoldenCross']].to_csv('cocacola_stock.csv')
    
if __name__ == '__main__':
    main()
