import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from src.data_import import data, data_2D
from src.MCMC_MH import run_mcmc_mh
from src.FEMBasis import FEMBasis2D

def plot_mcmc_results(data, B, samples, spline_basis, n_curves=50, seed=42):
    """
    Plot MCMC results including fitted curves and trace plots
    
    Parameters:
    -----------
    data : list of arrays
        Original data for each group
    B : array
        B-spline basis matrix
    samples : dict
        MCMC samples dictionary
    spline_basis : BSplineBasis
        B-spline basis object
    n_curves : int
        Number of posterior curves to plot
    seed : int
        Random seed for reproducibility
    """
    
    np.random.seed(seed)
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Fitted curves for first group
    ax1 = plt.subplot(2, 2, 1)
    c1, global_min, global_max = plot_fitted_curves(ax1, data, spline_basis, samples, B, group_idx=0, n_curves=n_curves)
    
    # 2. Original data scatter plot
    ax2 = plt.subplot(2, 2, 2)
    c2, _, _ = plot_data_scatter(ax2, data_2D, global_min, global_max)
    
    # 3. Trace plots for beta parameters
    ax3 = plt.subplot(2, 2, 3)
    plot_beta_traces(ax3, samples)
    
    # 4. Trace plots for variance parameters
    ax4 = plt.subplot(2, 2, 4)
    plot_variance_traces(ax4, samples)
    
    # Create shared colorbar for the contour plots
    plt.tight_layout()
    
    # Add a single colorbar for both contour plots
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    fig.colorbar(c1, cax=cbar_ax, label='Temp')
    
    plt.show()

def plot_fitted_curves(ax, data, fem, samples, B, group_idx=0, n_curves=1):
    """Plot fitted curves for a specific group

    Parameters:
    - ax: matplotlib axis
    - data: original data
    - fem: FEMBasis2D object (used to evaluate basis at points)
    - samples: MCMC samples dict
    - B: B-spline basis matrix (or other basis matrix) (kept for compatibility)
    
    Returns:
    - contour object, global_min, global_max
    """
    x = np.linspace(2, 22, 60)
    y = np.linspace(33, 53, 60)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    phi_plot = fem.evaluate_basis(points)  # fem should be a FEMBasis2D object

    beta = np.mean(samples['beta'], axis=0)
        
    curve = phi_plot @ beta
    curve_unstack = curve.reshape(X.shape)
    
    # Calculate data range for consistent color scaling
    data_averaged = np.mean(data_2D, axis=0)
    global_min = min(np.min(curve_unstack), np.min(data_averaged))
    global_max = max(np.max(curve_unstack), np.max(data_averaged))
    
    c = ax.contourf(X, Y, curve_unstack, levels=20, cmap='viridis', 
                    vmin=global_min, vmax=global_max)
    ax.scatter(fem.nodes[:, 0], fem.nodes[:, 1], c='k', s=20)  # show basis nodes
    ax.set_title('Fitted Surface (Posterior Mean over all betas)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    return c, global_min, global_max

def plot_data_scatter(ax, data_2D, global_min=None, global_max=None):
    """Plot original data as scatter plot
    
    Parameters:
    - ax: matplotlib axis
    - data_2D: 2D data array with shape (time_points, x_dim, y_dim)
    - global_min, global_max: optional global limits for consistent coloring
    
    Returns:
    - contour object, global_min, global_max
    """
    # Create coordinate grid matching data dimensions
    x_coords = np.linspace(2, 22, data_2D[0].shape[1])
    y_coords = np.linspace(33, 53, data_2D[0].shape[0])
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Skip first year and average the remaining data
    data_averaged = np.mean(data_2D, axis=0)
    
    # Use global limits if provided, otherwise calculate from this data
    if global_min is None or global_max is None:
        global_min = np.min(data_averaged)
        global_max = np.max(data_averaged)
    
    scatter = ax.contourf(X, Y, data_averaged, levels=20, cmap='viridis', 
                         vmin=global_min, vmax=global_max)
    
    ax.set_title('Original Data (Average over all years)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    return scatter, global_min, global_max

def plot_beta_traces(ax, samples):
    """Plot trace plots for beta parameters"""
    beta_samples = samples['beta']
    n_beta = beta_samples.shape[1]
    
    # Plot only first 10 beta parameters to avoid clutter
    n_to_plot = min(10, n_beta)
    for i in range(n_to_plot):
        ax.plot(beta_samples[:, i], alpha=0.7, label=f'β{i}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title(f'Trace Plots - Beta Parameters (first {n_to_plot})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_variance_traces(ax, samples):
    """Plot trace plots for variance parameters"""
    # Plot sigma_e
    ax.plot(samples['sigma_e'], alpha=0.7, label='σ_e', color='red')
    
    # If sigma_b is a matrix, plot its diagonal elements
    sigma_b_samples = samples['sigma_b']
    if sigma_b_samples.ndim == 3:
        n_b = sigma_b_samples.shape[1]
        # Plot first few diagonal elements
        n_to_plot = min(5, n_b)
        colors = plt.cm.Set1(np.linspace(0, 1, n_to_plot))
        for i in range(n_to_plot):
            ax.plot(sigma_b_samples[:, i, i], alpha=0.7, 
                   color=colors[i], label=f'σ_b[{i},{i}]')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Trace Plots - Variance Parameters')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Usage in your main script
if __name__ == "__main__":
  # --- 1. Create FEM Basis ---
    domain = ((2, 33), (22, 53))
    K = 64  # number of basis nodes
    fem = FEMBasis2D.from_domain(domain, K)
    x = np.linspace(2, 22, 20)
    y = np.linspace(33, 53, 20)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    phi = fem.evaluate_basis(points)    

    # --- 2. Import CSV data ---
    data_stack = [data_2D[i].reshape(phi.shape[0]) for i in range(len(data_2D))]

    # --- 3. Define Priors ---
    priors = {
        'c_beta': 100.0,            # Vague prior for beta
        'c_epsilon': 0.01,          # Vague prior for sigma_e
        'd_epsilon': 0.01,          # Vague prior for sigma_e
        'eta_b': K + 2,    # Prior df for sigma_b (min for defined mean)
        'S_b': np.eye(K)   # Prior mean matrix for sigma_b
    }
    
    # --- 4. Run MCMC ---
    start_time = perf_counter()
    samples = run_mcmc_mh(data_stack, phi, priors, n_iter=5000, n_burn=2500)
    print("\n--- Run Completed ---")
    elapsed = perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f}s")
    print(samples['beta'][0].shape)
    print(samples['b_0'].shape) 
    plot_mcmc_results(data, phi, samples, fem, n_curves=1)
    # print elapsed time
