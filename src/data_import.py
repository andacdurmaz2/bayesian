import pandas as pd
import os

def load_and_prepare_data(file_path="data/df_equator.csv"):
    """
    Load and prepare the data by filtering longitude and organizing by lon-year pairs.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with longitude as keys and DataFrames with year and mean as values
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["lon", "year"])
    
    # Filter longitude range
    df = df[(df["lon"] >= 30) & (df["lon"] < 60)]
    
    # Create dictionary with longitude as keys
    data = {
        lon: df_lon[["year", "mean"]].reset_index(drop=True)
        for lon, df_lon in df.groupby("lon")
    }
    
    return data

# Load data when module is imported
data = load_and_prepare_data()
