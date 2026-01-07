import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular, cholesky

def matern_covariance_32(distance_matrix, rho, sigma):
    """Compute the Matérn covariance matrix with v = 3/2."""
    # Add stability check
    if rho <= 0 or sigma <= 0:
        return np.eye(distance_matrix.shape[0]) * 1e-6
    
    # Compute the scaled distance with clipping
    scaled_dist = np.sqrt(3) * distance_matrix / np.clip(rho, 1e-8, None)
    
    # Compute the Matérn covariance with v = 3/2
    # Use np.exp with clipping to avoid overflow
    exp_term = np.exp(-np.clip(scaled_dist, None, 700))
    cov_matrix = sigma**2 * (1 + scaled_dist) * exp_term
    
    # Add small diagonal for numerical stability
    cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-8
    
    return cov_matrix

def distance_matrix(beta):
    """Create distance Matrix considering the box like strucure of the data"""
    dim = beta.shape[0]
    dim_sqrt = int(np.sqrt(dim))
    
    # Create coordinates using meshgrid
    x = np.arange(dim_sqrt)
    y = np.arange(dim_sqrt)
    X, Y = np.meshgrid(x, y)
    coords = np.column_stack((X.ravel(), Y.ravel()))
    
    # Use broadcasting 
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
    
    return dist_mat

def sigma_distribution_log(beta, b, rho, sgm, n, a_rho, b_rho, a_sgm, b_sgm):
    """
    Compute the LOG probability of the sigma_b defined. To avoid numerical overflow we use LOG.
    
    Returns: Probabibility for the given rho and sgm
    """
    # --- 1. Priors (work in log space) ---
    # Using scipy's logpdf directly
    roh_dist = stats.lognorm(s=b_rho, scale=np.exp(a_rho))
    sgm_dist = stats.lognorm(s=b_sgm, scale=np.exp(a_sgm))
    
    # Handle very small probability densities
    log_roh_prob = roh_dist.logpdf(rho)
    log_sgm_prob = sgm_dist.logpdf(sgm)
    
    # If PDF is 0, logpdf returns -inf, so we clip
    log_roh_prob = np.clip(log_roh_prob, -1e6, None)
    log_sgm_prob = np.clip(log_sgm_prob, -1e6, None)
    
    # --- 2. Create Covariance Matrix ---
    dist_mat = distance_matrix(beta)
    cov_mat = matern_covariance_32(dist_mat, rho, sgm)
    
    # --- 3. Cholesky decomposition for stability ---
    try:
        L = cholesky(cov_mat, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        # If Cholesky fails, matrix is not positive definite
        # Return very low probability
        return -np.inf
    
    # --- 4. Compute log determinant efficiently ---
    # log(det(cov)) = 2 * sum(log(diag(L)))
    log_det_cov = 2 * np.sum(np.log(np.diag(L)))
    
    # --- 5. Compute log-likelihood efficiently ---
    if beta.ndim == 1:
        beta_col = beta.reshape(-1, 1)
    else:
        beta_col = beta
    
    total_mahalanobis = 0.0
    
    for i in range(n):
        if b[i].ndim == 1:
            b_i_col = b[i].reshape(-1, 1)
        else:
            b_i_col = b[i]
        
        # Solve L*z = (b_i - beta) using forward substitution
        deviation = b_i_col - beta_col
        z = solve_triangular(L, deviation, lower=True, check_finite=False)
        
        # Mahalanobis distance squared = z^T * z
        total_mahalanobis += float(z.T @ z)
    
    # --- 6. Combine all log probabilities ---
    # Original formula: prob = det(cov)^(-n/2) * exp(-sum_of_squares/2) * priors
    # In log space: log_prob = (-n/2)*log(det(cov)) - sum_of_squares/2 + log(priors)
    
    log_likelihood = (-n/2) * log_det_cov - total_mahalanobis/2
    
    # Total log probability
    log_prob = log_likelihood + log_roh_prob + log_sgm_prob
    
    # Additional stability: if log_prob is too small, return a very small number
    if not np.isfinite(log_prob):
        return -1e6
    
    return float(log_prob)

# For backward compatibility, you can keep the original but use log version internally
def sigma_distribution(beta, b, rho, sgm, n, a_rho, b_rho, a_sgm, b_sgm):
    """Original function using safe computation."""
    log_prob = sigma_distribution_log(beta, b, rho, sgm, n, a_rho, b_rho, a_sgm, b_sgm)
    
    # Only exponentiate if needed (be careful!)
    if log_prob < -700:
        return 0.0
    else:
        return np.exp(log_prob)