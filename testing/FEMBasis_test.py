import numpy as np
import matplotlib.pyplot as plt
from src.FEMBasis import FEMBasis2D

# --- Create FEMBasis2D instance using K ---
domain = ((2, 33), (22, 53))
K = 20  # number of basis nodes
fem = FEMBasis2D.from_domain(domain, K)

# --- Create grid of evaluation points ---
x = np.linspace(2, 22, 100)
y = np.linspace(33, 53, 100)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T

# --- Evaluate basis functions at grid points ---
phi = fem.evaluate_basis(points)

# --- Plot first 4 basis functions (or fewer if less nodes) ---
num_plots = min(4, len(fem))  # first 4 nodes
fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))

if num_plots == 1:
    axes = [axes]

for i in range(num_plots):
    Z = phi[:, i].reshape(X.shape)
    ax = axes[i]
    c = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.scatter(fem.nodes[:, 0], fem.nodes[:, 1], c='k', s=20)  # show basis nodes
    ax.set_title(f'Basis at node {i}')
    fig.colorbar(c, ax=ax)

plt.tight_layout()
plt.show()
