import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for nicer plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def check_files_exist():
    """Check if required files exist."""
    required_files = [
        'results/phi_matrix.npy',
        'results/samples_inverse_wishart.pkl',
        'results/samples_metropolis_hastings.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run 'python sampling_script.py' first!")
        return False
    
    return True

def load_samples(filename):
    """Load samples from pickle file."""
    filepath = f'results/{filename}.pkl'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_covariance_matrix(cov_matrix_1, title="Covariance Matrix", 
                          figsize=(12, 10), cmap="RdBu_r",
                          colorbar=True, fontsize=12, dpi=100):
    """
    Plot a covariance matrix with nice formatting.
    """
    cov_matrix_1 = np.asarray(cov_matrix_1)
    
    # Check if we need to take mean over samples
    if len(cov_matrix_1.shape) == 3:
        print(f"  Taking mean over {cov_matrix_1.shape[0]} samples")
        cov_matrix_1 = np.mean(cov_matrix_1, axis=0)

    cov_matrix=np.linalg.inv(cov_matrix_1)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create symmetric colormap centered at 0
    vmax = np.max(np.abs(cov_matrix))
    vmin = -vmax
    
    # Create heatmap
    im = ax.imshow(cov_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                   interpolation='nearest', aspect='auto')
    
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Covariance', fontsize=fontsize)
    
    # Remove ticks for large matrices
    if cov_matrix.shape[0] > 50:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(labelsize=fontsize-2)
    
    plt.tight_layout()
    return fig, ax

def plot_covariance_evolution(sigma_samples, method_name, n_samples_to_show=5, save_path=None):
    """
    Plot evolution of covariance matrix over samples.
    """
    n_samples_total = sigma_samples.shape[0]
    
    # Select evenly spaced samples
    indices = np.linspace(0, n_samples_total-1, n_samples_to_show, dtype=int)
    
    fig, axes = plt.subplots(1, n_samples_to_show, figsize=(5*n_samples_to_show, 5))
    
    if n_samples_to_show == 1:
        axes = [axes]
    
    for idx, (sample_idx, ax) in enumerate(zip(indices, axes)):
        cov_matrix = sigma_samples[sample_idx]
        
        # Create symmetric colormap
        vmax = np.max(np.abs(cov_matrix))
        vmin = -vmax
        
        im = ax.imshow(cov_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                      interpolation='nearest', aspect='auto')
        
        ax.set_title(f'Sample {sample_idx}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f'Covariance Matrix Evolution - {method_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_beta_comparison(beta_iw, beta_mh, save_path=None, title='no title spec'):
    """
    Plot comparison of beta coefficients from IW and MH.
    """
    K = len(beta_iw)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of MCMC Sampling Methods', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot beta IW
    ax = axes[0, 0]
    x_ax = np.arange(K)
    ax.scatter(x_ax, beta_iw, alpha=0.6, s=30, label='IW samples', color='blue')
    ax.plot(x_ax, beta_iw, 'b-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Coefficient Index', fontsize=11)
    ax.set_ylabel('IW Value', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot beta MH
    ax = axes[0, 1]
    ax.scatter(x_ax, beta_mh, alpha=0.6, s=30, color='orange', label='MH samples')
    ax.plot(x_ax, beta_mh, 'orange', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Coefficient Index', fontsize=11)
    ax.set_ylabel('MH Value', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot difference
    ax = axes[1, 0]
    beta_diff = beta_iw - beta_mh
    ax.scatter(x_ax, beta_diff, alpha=0.6, s=30, color='green', label='Difference')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.fill_between(x_ax, -np.std(beta_diff), np.std(beta_diff), 
                    alpha=0.1, color='gray', label='±1 std')
    ax.set_xlabel('Coefficient Index', fontsize=11)
    ax.set_ylabel('Difference (IW - MH)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot histogram of differences
    ax = axes[1, 1]
    ax.hist(beta_diff, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Zero difference')
    ax.axvline(x=np.mean(beta_diff), color='b', linestyle='-', alpha=0.7, 
               linewidth=2, label=f'Mean: {np.mean(beta_diff):.4f}')
    ax.set_xlabel('Difference', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_trace_plots(samples, method_name, max_params=9, save_path=None):
    """
    Plot trace plots for key parameters.
    """
    if 'beta' not in samples:
        print(f"No beta samples found for {method_name}")
        return None
    
    beta_samples = samples['beta']
    
    # Check if beta_samples is 2D
    if len(beta_samples.shape) != 2:
        print(f"Beta samples for {method_name} are not 2D (shape: {beta_samples.shape})")
        return None
    
    n_samples, n_params_total = beta_samples.shape
    n_params = min(max_params, n_params_total)  # Plot at most max_params parameters
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for i in range(n_params):
        ax = axes_flat[i]
        ax.plot(beta_samples[:, i], alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel(f'β{i+1}', fontsize=10)
        ax.set_title(f'Trace Plot for β{i+1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].axis('off')
    
    fig.suptitle(f'Trace Plots - {method_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_summary_statistics(samples_iw, samples_mh):
    """
    Create summary statistics for both methods.
    """
    stats = {}
    
    # Beta statistics
    if 'beta' in samples_iw and 'beta' in samples_mh:
        beta_iw_mean = np.mean(samples_iw['beta'], axis=0)
        beta_mh_mean = np.mean(samples_mh['beta'], axis=0)
        
        stats['beta_iw_mean'] = np.mean(beta_iw_mean)
        stats['beta_iw_std'] = np.std(beta_iw_mean)
        stats['beta_mh_mean'] = np.mean(beta_mh_mean)
        stats['beta_mh_std'] = np.std(beta_mh_mean)
        stats['beta_diff_mean'] = np.mean(beta_iw_mean - beta_mh_mean)
        stats['beta_diff_std'] = np.std(beta_iw_mean - beta_mh_mean)
        stats['beta_correlation'] = np.corrcoef(beta_iw_mean, beta_mh_mean)[0, 1]
    
    # Sigma_b statistics
    if 'sigma_b' in samples_iw and 'sigma_b' in samples_mh:
        # Take mean over samples to get average covariance matrix
        sigma_iw = np.mean(samples_iw['sigma_b'], axis=0)
        sigma_mh = np.mean(samples_mh['sigma_b'], axis=0)
        
        stats['sigma_iw_norm'] = np.linalg.norm(sigma_iw)
        stats['sigma_mh_norm'] = np.linalg.norm(sigma_mh)
        stats['sigma_diff_norm'] = np.linalg.norm(sigma_iw - sigma_mh)
        
        # Eigenvalue comparison
        eig_iw = np.linalg.eigvalsh(sigma_iw)
        eig_mh = np.linalg.eigvalsh(sigma_mh)
        stats['eig_iw_mean'] = np.mean(eig_iw)
        stats['eig_mh_mean'] = np.mean(eig_mh)
        stats['eig_iw_max'] = np.max(eig_iw)
        stats['eig_mh_max'] = np.max(eig_mh)
        stats['eig_iw_min'] = np.min(eig_iw)
        stats['eig_mh_min'] = np.min(eig_mh)
        
        # Also compute statistics across samples
        sigma_iw_std = np.std(samples_iw['sigma_b'], axis=0)
        sigma_mh_std = np.std(samples_mh['sigma_b'], axis=0)
        stats['sigma_iw_std_norm'] = np.linalg.norm(sigma_iw_std)
        stats['sigma_mh_std_norm'] = np.linalg.norm(sigma_mh_std)
    
    return stats

def plot_covariance_statistics(sigma_iw_samples, sigma_mh_samples, save_dir='results/plots'):
    """
    Plot statistics of covariance matrices across samples.
    """
    # Compute statistics
    sigma_iw_mean = np.mean(sigma_iw_samples, axis=0)
    sigma_mh_mean = np.mean(sigma_mh_samples, axis=0)
    sigma_iw_std = np.std(sigma_iw_samples, axis=0)
    sigma_mh_std = np.std(sigma_mh_samples, axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Mean covariance matrices
    vmax1 = max(np.max(np.abs(sigma_iw_mean)), np.max(np.abs(sigma_mh_mean)))
    vmin1 = -vmax1
    
    ax = axes[0, 0]
    im1 = ax.imshow(sigma_iw_mean, cmap="RdBu_r", vmin=vmin1, vmax=vmax1)
    ax.set_title('Mean Covariance - IW', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im1, ax=ax, shrink=0.8)
    
    ax = axes[0, 1]
    im2 = ax.imshow(sigma_mh_mean, cmap="RdBu_r", vmin=vmin1, vmax=vmax1)
    ax.set_title('Mean Covariance - MH', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im2, ax=ax, shrink=0.8)
    
    ax = axes[0, 2]
    diff_mean = sigma_iw_mean - sigma_mh_mean
    vmax2 = np.max(np.abs(diff_mean))
    vmin2 = -vmax2
    im3 = ax.imshow(diff_mean, cmap="RdBu_r", vmin=vmin2, vmax=vmax2)
    ax.set_title('Difference in Mean Covariance', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im3, ax=ax, shrink=0.8)
    
    # Standard deviation matrices
    vmax3 = max(np.max(sigma_iw_std), np.max(sigma_mh_std))
    
    ax = axes[1, 0]
    im4 = ax.imshow(sigma_iw_std, cmap="viridis", vmin=0, vmax=vmax3)
    ax.set_title('Std Dev of Covariance - IW', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im4, ax=ax, shrink=0.8)
    
    ax = axes[1, 1]
    im5 = ax.imshow(sigma_mh_std, cmap="viridis", vmin=0, vmax=vmax3)
    ax.set_title('Std Dev of Covariance - MH', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im5, ax=ax, shrink=0.8)
    
    ax = axes[1, 2]
    diff_std = sigma_iw_std - sigma_mh_std
    vmax4 = np.max(np.abs(diff_std))
    vmin4 = -vmax4
    im6 = ax.imshow(diff_std, cmap="RdBu_r", vmin=vmin4, vmax=vmax4)
    ax.set_title('Difference in Std Dev', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im6, ax=ax, shrink=0.8)
    
    plt.suptitle('Covariance Matrix Statistics Across Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'covariance_statistics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def main():
    print("=" * 60)
    print("MCMC Results Visualization")
    print("=" * 60)
    
    # Check if files exist
    if not check_files_exist():
        return
    
    # Create plots directory
    plots_dir = 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nLoading saved samples...")
    
    try:
        # Load samples
        samples_iw = load_samples('samples_inverse_wishart')
        samples_mh = load_samples('samples_metropolis_hastings')
        
        print("✓ Successfully loaded samples")
        
        # Get beta means
        beta_iw_mean = np.mean(samples_iw['beta'], axis=0)
        beta_mh_mean = np.mean(samples_mh['beta'], axis=0)
        K = len(beta_iw_mean)
        
        # Get b mean
        b_iw_mean=np.mean(samples_iw['b_0'],axis=0)
        b_mh_mean=np.mean(samples_mh['b_0'],axis=0)
        
        print(f"\nSample shapes:")
        print(f"  IW beta: {samples_iw['beta'].shape}")
        print(f"  MH beta: {samples_mh['beta'].shape}")
        print(f"  IW sigma_b: {samples_iw['sigma_b'].shape}")
        print(f"  MH sigma_b: {samples_mh['sigma_b'].shape}")
        print(f"  IW b: {samples_iw['b_0'].shape}")
        print(f"  MH b: {samples_mh['b_0'].shape}")
        print(f"  IW b_mean: {b_iw_mean.shape}")
        print(f"  MH b_mean: {b_mh_mean.shape}")
        print("\nCreating plots...")
        
        # 1. Beta comparison plot
        print("1. Creating beta comparison plot...")
        fig_beta = plot_beta_comparison(beta_iw_mean, beta_mh_mean,
                                       save_path=f'{plots_dir}/beta_comparison.png',title='Beta')
        plt.show()

        print("1.5 Creating b_avg comparison plot...")
        beta_normal_avg=np.mean(b_iw_mean,axis=0)
        beta_mh_avg=np.mean(b_mh_mean,axis=0)
        print(beta_mh_avg.shape)
        fig_beta = plot_beta_comparison(beta_normal_avg, beta_mh_avg,
                                       save_path=f'{plots_dir}/beta_comparison.png',title='avg all b_i')
        plt.show()
        
        
        # 2. Trace plots
        print("2. Creating b comparison plots...")
        for i in range(b_mh_mean.shape[0]):
            fig_beta = plot_beta_comparison(b_iw_mean[i], b_mh_mean[i],
                                       save_path=f'{plots_dir}/beta_comparison.png',title=f'b_{i}')
            plt.show()
        
        # 3. Covariance matrices (mean over samples)
        print("3. Creating covariance matrix plots...")
        
        if 'sigma_b' in samples_iw:
            fig, _ = plot_covariance_matrix(
                samples_iw['sigma_b'],
                title="Mean Covariance Matrix - Inverse Wishart",
                figsize=(12, 10)
            )
            plt.show()
        
        if 'sigma_b' in samples_mh:
            fig, _ = plot_covariance_matrix(
                samples_mh['sigma_b'],
                title="Mean Covariance Matrix - Metropolis-Hastings",
                figsize=(12, 10)
            )
            plt.show()
        
        if 'sigma_b' in samples_iw and 'sigma_b' in samples_mh:
            # Get mean covariance matrices
            sigma_iw_mean = np.mean(samples_iw['sigma_b'], axis=0)
            sigma_mh_mean = np.linalg.inv(np.mean(samples_mh['sigma_b'], axis=0))
            sigma_diff = sigma_iw_mean - sigma_mh_mean
            
            fig, _ = plot_covariance_matrix(
                sigma_diff,
                title="Difference in Mean Covariance Matrices (IW - MH)",
                figsize=(12, 10)
            )
            plt.show()
        
        # 4. Covariance evolution plots
        print("4. Creating covariance evolution plots...")
        if 'sigma_b' in samples_iw:
            fig_evol_iw = plot_covariance_evolution(
                samples_iw['sigma_b'],
                'Inverse Wishart',
                n_samples_to_show=5,
                save_path=f'{plots_dir}/covariance_evolution_iw.png'
            )
            if fig_evol_iw:
                plt.figure(fig_evol_iw.number)
                plt.show()
        
        if 'sigma_b' in samples_mh:
            fig_evol_mh = plot_covariance_evolution(
                samples_mh['sigma_b'],
                'Metropolis-Hastings',
                n_samples_to_show=5,
                save_path=f'{plots_dir}/covariance_evolution_mh.png'
            )
            if fig_evol_mh:
                plt.figure(fig_evol_mh.number)
                plt.show()
        
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()