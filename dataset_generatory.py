import numpy as np
import pandas as pd

def save_sin_to_excel(filename="sin_data.xlsx"):
    """
    Generates the first 10,000 points of the sine function and saves them to an Excel sheet.
    
    Args:
        filename (str): The name of the Excel file to save the data.
    """
    # Generate x values (0 to 9999)
    x_values = np.arange(10000)
    
    # Compute the sine values
    sin_values = np.sin(x_values)
    
    # Create a DataFrame
    data = pd.DataFrame({
        "X": x_values,
        "Sin(X)": sin_values
    })
    
    # Save to Excel
    data.to_excel(filename, index=False, engine="openpyxl")
    print(f"Data saved to {filename}")

# Call the function to save data
save_sin_to_excel()
