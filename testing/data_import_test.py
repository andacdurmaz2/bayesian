import matplotlib.pyplot as plt
import numpy as np
from src.data_import import data

def plot_longitude_means(data_dict):
    """
    Efficiently plot mean values for each longitude across years.
    
    Args:
        data_dict (dict): Dictionary containing longitude DataFrames
    """
    plt.figure(figsize=(10, 6))
    
    # Get all unique years from the dataset
    all_years = sorted(set().union(*[df['year'].unique() for df in data_dict.values()]))
    
    # Create color mapping
    cmap = plt.get_cmap('tab20')
    year_colors = {year: cmap(i % 20) for i, year in enumerate(all_years)}
    
    # Plot data efficiently using vectorized operations
    for i, (lon, df_lon) in enumerate(data_dict.items()):
        for year in all_years:
            year_data = df_lon[df_lon['year'] == year]['mean']
            if not year_data.empty:
                # Only add label for first longitude to avoid duplicate legend entries
                label = str(year) if i == 0 else ""
                plt.scatter([lon] * len(year_data), year_data, 
                           color=year_colors[year], label=label, alpha=0.7)
    
    plt.title("Mean per Year for Each Longitude")
    plt.xlabel("Longitude")
    plt.ylabel("Mean")
    plt.grid(True, alpha=0.3)
    
    # Create clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles = [handles[all_years.index(int(label))] for label in sorted(set(labels), key=int)]
    plt.legend(unique_handles, sorted(set(labels), key=int), title="Year")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_longitude_means(data)
    print('Plot generated successfully!')