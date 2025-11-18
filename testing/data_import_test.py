import matplotlib.pyplot as plt
import numpy as np
from src.data_import import data

def plot_longitude_means(data_list, years=None):
    """
    Efficiently plot mean values for each longitude across years.
    
    Args:
        data_list (list): List of numpy arrays where each array contains temperatures for all longitudes at a specific time
        years (list): Optional list of years corresponding to each time point
    """
    plt.figure(figsize=(10, 6))
    
    # Convert list to numpy array for easier manipulation
    data_array = np.array(data_list)  # Shape: (time_points, longitudes)
    
    # If years not provided, create sequential indices
    if years is None:
        years = list(range(len(data_array)))
    
    # Get unique longitudes (assuming they're sequential from 30-60)
    n_longitudes = data_array.shape[1]
    longitudes = np.linspace(30, 60, n_longitudes, endpoint=False)
    
    # Create color mapping
    cmap = plt.get_cmap('tab20')
    year_colors = {year: cmap(i % 20) for i, year in enumerate(years)}
    
    # Plot data efficiently - for each time point, plot all longitudes
    for time_idx, year in enumerate(years):
        # Get temperatures for all longitudes at this time point
        temps = data_array[time_idx]
        
        # Only add label for first time point to avoid duplicate legend entries
        label = str(year) if time_idx == 0 else ""
        
        # Plot all longitudes for this time point
        plt.scatter(longitudes, temps, 
                   color=year_colors[year], label=label, alpha=0.7, s=20)
    
    plt.title("Mean Temperature per Year for Each Longitude")
    plt.xlabel("Longitude")
    plt.ylabel("Mean Temperature")
    plt.grid(True, alpha=0.3)
    
    # Create clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:  # Only create legend if there are handles
        plt.legend(handles, labels, title="Year", loc='best')
    
    plt.tight_layout()
    plt.show()

# Alternative version that plots each longitude-year pair individually (like your original)
def plot_longitude_means_individual(data_list, years=None):
    """
    Plot each longitude-year pair individually, mimicking the original behavior.
    """
    plt.figure(figsize=(10, 6))
    
    data_array = np.array(data_list)
    n_longitudes = data_array.shape[1]
    n_times = data_array.shape[0]
    
    # Create longitude values
    longitudes = np.linspace(30, 60, n_longitudes, endpoint=False)
    
    if years is None:
        years = list(range(n_times))
    
    # Create color mapping
    cmap = plt.get_cmap('tab20')
    year_colors = {year: cmap(i % 20) for i, year in enumerate(years)}
    
    # Plot each longitude and time point individually
    for lon_idx, lon in enumerate(longitudes):
        for time_idx, year in enumerate(years):
            temp = data_array[time_idx, lon_idx]
            # Only add label for first longitude to avoid duplicate legend entries
            label = str(year) if lon_idx == 0 else ""
            plt.scatter(lon, temp, 
                       color=year_colors[year], label=label, alpha=0.7, s=20)
    
    plt.title("Mean Temperature per Year for Each Longitude")
    plt.xlabel("Longitude")
    plt.ylabel("Mean Temperature")
    plt.grid(True, alpha=0.3)
    
    # Create clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique and label:  # Only consider non-empty labels
            unique[label] = handle
    
    # Sort years numerically
    sorted_years = sorted(unique.keys(), key=int)
    unique_handles = [unique[year] for year in sorted_years]
    
    plt.legend(unique_handles, sorted_years, title="Year", loc='best')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # First, let's debug the data structure
    print(f"Data type: {type(data)}")
    print(f"Data length: {len(data)}")
    if len(data) > 0:
        print(f"First element type: {type(data[0])}")
        print(f"First element shape: {data[0].shape}")
        print(f"First few longitudes sample: {data[0][:5]}")  # First 5 values
    
    # Test with the individual plotting version (more similar to original)
    plot_longitude_means_individual(data)
    print('Plot generated successfully!')