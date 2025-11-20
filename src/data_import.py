import pandas as pd
import numpy as np
import os

def load_and_prepare_data(file_path="data/df_equator.csv"):
    """
    Load and prepare the data by filtering longitude and organizing by time-longitude pairs.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        list: List of numpy arrays where each array contains temperatures for all longitudes at a specific time
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["lon", "year"])
    
    # Filter longitude range
    df = df[(df["lon"] >= 30) & (df["lon"] < 60)]
    
    # Pivot the dataframe to have years as rows and longitudes as columns
    pivot_df = df.pivot(index='year', columns='lon', values='mean')
    
    # Create the desired structure: list of numpy arrays
    # Each array represents temperatures across all longitudes for a specific year
    data = [pivot_df.iloc[i].values for i in range(len(pivot_df))]
    
    return data

# Alternative version if you want to preserve year information:
def load_and_prepare_data_with_years(file_path="data/df_equator.csv"):
    """
    Load and prepare the data, returning both the data structure and years.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (data, years) where data is list of numpy arrays and years is the corresponding years
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["lon", "year"])
    
    # Filter longitude range
    df = df[(df["lon"] >= 30) & (df["lon"] < 60)]
    
    # Pivot the dataframe
    pivot_df = df.pivot(index='year', columns='lon', values='mean')
    
    # Get years for reference
    years = pivot_df.index.tolist()
    
    # Create the data structure
    data = [pivot_df.iloc[i].values for i in range(len(pivot_df))]
    
    return data, years

# Load data when module is imported
data = load_and_prepare_data()

# If you need the years as well:
# data, years = load_and_prepare_data_with_years()