import numpy as np
from scipy import stats
from src.sigma_distribution import sigma_distribution, distance_matrix,matern_covariance_32
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import MaxNLocator

beta=np.ones(4)
b=np.zeros((10,4))
rho=1
sgm=1
a_rho=0.5
b_rho=0.1
a_sgm=1
b_sgm=0.5
n=10



# Create a grid of rho and sgm values
rho_vals = np.linspace(1, 5, 100)  # Adjust range as needed
sgm_vals = np.linspace(0.01, 1.5, 100)  # Adjust range as needed
rho_grid, sgm_grid = np.meshgrid(rho_vals, sgm_vals)

# Calculate the function values
z_values = np.zeros_like(rho_grid)
for i in range(len(rho_vals)):
    for j in range(len(sgm_vals)):
        z_values[i, j] = sigma_distribution(
            beta, b, rho_grid[i, j], sgm_grid[i, j], 
            n, a_rho, b_rho, a_sgm, b_sgm
        )

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(rho_grid, sgm_grid, z_values, 
                       cmap='viridis', 
                       alpha=0.8,
                       linewidth=0,
                       antialiased=True)

# Add contour lines on the axes
ax.contour(rho_grid, sgm_grid, z_values, zdir='z', offset=z_values.min(), 
           cmap='coolwarm', alpha=0.5)

# Add labels
ax.set_xlabel('rho', fontsize=12, labelpad=10)
ax.set_ylabel('sgm', fontsize=12, labelpad=10)
ax.set_zlabel('sigma_distribution', fontsize=12, labelpad=10)
ax.set_title('3D Plot of sigma_distribution function', fontsize=14, pad=20)

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Function Value')

# Adjust viewing angle
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# Optional: Also create a heatmap/2D projection
fig2, ax2 = plt.subplots(figsize=(10, 8))
contour = ax2.contourf(rho_grid, sgm_grid, z_values, levels=50, cmap='viridis')
ax2.set_xlabel('rho', fontsize=12)
ax2.set_ylabel('sgm', fontsize=12)
ax2.set_title('Heatmap of sigma_distribution', fontsize=14)
fig2.colorbar(contour, ax=ax2, label='Function Value')
plt.tight_layout()
plt.show()


def plot_covariance_matrix(cov_matrix, title="64×64 Covariance Matrix", 
                          annot=False, figsize=(14, 12), cmap="RdBu_r",
                          colorbar=True, show_diagonal=True, 
                          fontsize=10, dpi=100, 
                          grid_lines=False, show_ticks=True):
    """
    Plot a 64x64 covariance matrix with nice formatting.
    
    Parameters
    ----------
    cov_matrix : array-like, shape (64, 64)
        The covariance matrix to plot
    title : str, optional
        Title for the plot
    annot : bool, optional
        Whether to annotate cells with values (not recommended for 64x64)
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap for the heatmap
    colorbar : bool, optional
        Whether to show colorbar
    show_diagonal : bool, optional
        Whether to highlight the diagonal
    fontsize : int, optional
        Font size for title and labels
    dpi : int, optional
        Figure resolution
    grid_lines : bool, optional
        Whether to show grid lines
    show_ticks : bool, optional
        Whether to show tick labels
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Convert to numpy array if not already
    cov_matrix = np.asarray(cov_matrix)
    
    # Validate shape
    if cov_matrix.shape != (64, 64):
        raise ValueError(f"Expected shape (64, 64), got {cov_matrix.shape}")
    
    # Create figure with higher DPI for clarity
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # For large matrices, disable annotations by default
    if annot and cov_matrix.size > 100:
        print("Warning: Annotations may be unreadable for large matrices. Consider setting annot=False.")
    
    # Create symmetric colormap centered at 0 for covariance
    if cmap in ["RdBu_r", "coolwarm", "bwr", "seismic"]:
        # Find absolute maximum for symmetric colormap
        vmax = np.max(np.abs(cov_matrix))
        vmin = -vmax
        center = 0
    else:
        vmin, vmax = None, None
        center = None
    
    # Create heatmap
    heatmap = sns.heatmap(cov_matrix,
                ax=ax,
                cmap=cmap,
                annot=annot,
                fmt=".2f" if annot else None,
                annot_kws={"size": 4},  # Very small for large matrix
                square=True,
                cbar=colorbar,
                cbar_kws={'shrink': 0.8, 'label': 'Covariance'},
                vmin=vmin,
                vmax=vmax,
                center=center,
                linewidths=0 if not grid_lines else 0.2,
                linecolor='gray' if grid_lines else 'white',
                xticklabels=show_ticks,
                yticklabels=show_ticks)
    
    # Customize plot
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
    
    # Show only every Nth tick for readability
    if show_ticks:
        n_ticks = 16  # Show 16 ticks (every 4th)
        tick_positions = np.linspace(0, 63, n_ticks, dtype=int)
        tick_labels = [str(i+1) for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=fontsize-2)
        ax.set_yticklabels(tick_labels, fontsize=fontsize-2)
        
        # Add axis labels
        ax.set_xlabel('Variable Index', fontsize=fontsize, labelpad=10)
        ax.set_ylabel('Variable Index', fontsize=fontsize, labelpad=10)
    else:
        # Remove tick labels but keep axes
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Highlight diagonal if requested
    if show_diagonal:
        # Add a line along the diagonal
        ax.plot([0, 64], [0, 64], color='black', linewidth=1, alpha=0.7, transform=ax.transData)
        # Or highlight with a different color using fill_between
        # ax.fill_between(range(65), range(65), color='yellow', alpha=0.1)
    
    # Add grid for better orientation (optional)
    if grid_lines:
        ax.set_xticks(np.arange(64) + 0.5, minor=True)
        ax.set_yticks(np.arange(64) + 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2, alpha=0.5)
        ax.tick_params(which="minor", length=0)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def plot_covariance_matrix_2(cov_matrix, title="16×16 Covariance Matrix", 
                          annot=True, figsize=(10, 8), cmap="RdBu_r",
                          colorbar=True, show_diagonal=True, 
                          fontsize=12, dpi=100, 
                          grid_lines=True, show_ticks=True,
                          annot_fmt=".3f", annot_size=10):
    """
    Plot a 5x5 covariance matrix with nice formatting.
    
    Parameters
    ----------
    cov_matrix : array-like, shape (5, 5)
        The covariance matrix to plot
    title : str, optional
        Title for the plot
    annot : bool, optional
        Whether to annotate cells with values
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap for the heatmap
    colorbar : bool, optional
        Whether to show colorbar
    show_diagonal : bool, optional
        Whether to highlight the diagonal
    fontsize : int, optional
        Font size for title and labels
    dpi : int, optional
        Figure resolution
    grid_lines : bool, optional
        Whether to show grid lines
    show_ticks : bool, optional
        Whether to show tick labels
    annot_fmt : str, optional
        Format string for annotations
    annot_size : int, optional
        Font size for annotations
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Convert to numpy array if not already
    cov_matrix = np.asarray(cov_matrix)
    
    # Validate shape
    if cov_matrix.shape != (16, 16):
        raise ValueError(f"Expected shape (5, 5), got {cov_matrix.shape}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create symmetric colormap centered at 0 for covariance
    if cmap in ["RdBu_r", "coolwarm", "bwr", "seismic"]:
        # Find absolute maximum for symmetric colormap
        vmax = np.max(np.abs(cov_matrix))
        vmin = -vmax
        center = 0
    else:
        vmin, vmax = None, None
        center = None
    
    # Create heatmap
    heatmap = sns.heatmap(cov_matrix,
                ax=ax,
                cmap=cmap,
                annot=annot,
                fmt=annot_fmt if annot else None,
                annot_kws={"size": annot_size, "weight": "bold"},
                square=True,
                cbar=colorbar,
                cbar_kws={'shrink': 0.8, 'label': 'Covariance'},
                vmin=vmin,
                vmax=vmax,
                center=center,
                linewidths=1 if grid_lines else 0,
                linecolor='black' if grid_lines else 'white',
                xticklabels=show_ticks,
                yticklabels=show_ticks)
    
    # Customize plot
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
    
    # Set tick labels
    if show_ticks:
        # For 5x5 matrix, we can show all ticks clearly
        tick_labels = [f'b{i+1}' for i in range(16)]
        ax.set_xticklabels(tick_labels, fontsize=fontsize, rotation=0)
        ax.set_yticklabels(tick_labels, fontsize=fontsize, rotation=0)
        
        # Add axis labels
        ax.set_xlabel('random coef: b_i', fontsize=fontsize, labelpad=10)
        ax.set_ylabel('random coef: b_i', fontsize=fontsize, labelpad=10)
    else:
        # Remove tick labels but keep axes
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Highlight diagonal if requested
    if show_diagonal:
        # Add thick line along the diagonal
        ax.plot([0, 16], [0, 16], color='black', linewidth=2, alpha=0.8, transform=ax.transData)
        
        # Alternatively, highlight diagonal cells
        # for i in range(5):
        #     rect = plt.Rectangle((i, i), 1, 1, fill=False, 
        #                          edgecolor='yellow', linewidth=2)
        #     ax.add_patch(rect)
    
    # Add thicker grid for better visibility (for 5x5)
    if grid_lines:
        ax.set_xticks(np.arange(16) + 0.5, minor=True)
        ax.set_yticks(np.arange(16) + 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5, alpha=0.5)
        ax.tick_params(which="minor", length=0)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax
beta=np.ones(16)
l=np.ones(16)
cov_matrix=matern_covariance_32(distance_matrix(l),rho,sgm)
print(cov_matrix.shape)
fig1, ax1 = plot_covariance_matrix_2(
        cov_matrix,
        title="16 x 16 Matérn Covariance Matrix with rho = sgm = 1",
        annot=False,  # Too dense for 64x64
        cmap="RdBu_r",
        show_ticks=True,
        grid_lines=False
    )
plt.show()
