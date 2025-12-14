import numpy as np
from scipy import stats
from src.sigma_distribution import sigma_distribution
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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