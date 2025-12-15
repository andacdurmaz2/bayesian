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
    data = [pivot_df.iloc[i].values for i in range(len(pivot_df)-20)]
    
    return data

def load_and_prepare_data_2D(file_path="data/df_equator_2D.csv"):
    """
    Load and prepare the data by filtering latitude (33-53) and longitude (2-22) ranges.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        list: List of 2D numpy arrays where each array contains temperatures for all latitude-longitude pairs at a specific year
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["year", "lat", "lon"])
    
    # Filter latitude and longitude ranges
    df = df[(df["lat"] >= 33) & (df["lat"] < 53) & 
            (df["lon"] >= 2) & (df["lon"] < 22)]
    
    # Get unique coordinates to understand the grid structure
    lats = sorted(df['lat'].unique())
    lons = sorted(df['lon'].unique())
    years = sorted(df['year'].unique())
    
    print(f"Grid dimensions: {len(lats)} latitudes × {len(lons)} longitudes")
    print(f"Number of years: {len(years)}")
    print(f"Latitude range: {min(lats)} to {max(lats)}")
    print(f"Longitude range: {min(lons)} to {max(lons)}")
    
    # Create the 2D structure: list of 2D arrays (lat × lon) for each year
    data = []
    
    years=years[-4:]
    print('Number of years:',len(years))
    for year in years:
        year_data = df[df['year'] == year]
        
        # Pivot to create 2D grid: rows=latitude, columns=longitude
        pivot_2d = year_data.pivot(index='lat', columns='lon', values='mean')
        
        # Reindex to ensure consistent shape across all years
        # This fills in any missing lat/lon combinations with NaN
        pivot_2d = pivot_2d.reindex(index=lats, columns=lons)
        
        data.append(pivot_2d.values)
    
    return data
# Example usage:
# data, years, lats, lons = extract_temperature_3d("your_file.css")
# print(f"Number of years: {len(data)}")
# print(f"Matrix shape for each year: {data[0].shape} (latitudes × longitudes)")
# print(f"Years: {years}")
# print(f"Latitudes: {lats}")
# print(f"Longitudes: {lons}")
data_2D = load_and_prepare_data_2D()      #<---- This is 2D
data = load_and_prepare_data()      #<---- This is 1D
