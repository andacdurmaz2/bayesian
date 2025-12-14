import numpy as np
from scipy import stats

def matern_covariance_32(distance_matrix, rho, sigma):
    """
    Compute the Matérn covariance matrix with v = 3/2.
    
    Parameters:
    -----------
    distance_matrix : numpy.ndarray
        Matrix of pairwise distances between points
    rho : float
        Length-scale parameter (controls how quickly correlation decays)
    sigma : float
        Variance parameter (overall scale of the covariance)
    
    Returns:
    --------
    numpy.ndarray
        Matérn covariance matrix
    
    Notes:
    ------
    The Matérn covariance function with v = 3/2 is:
    C(d) = sigma^2 * (1 + sqrt(3)*d/rho) * exp(-sqrt(3)*d/rho)
    where d is the distance between two points.
    """
    # Compute the scaled distance
    scaled_dist = np.sqrt(3) * distance_matrix / rho
    
    # Compute the Matérn covariance with v = 3/2
    cov_matrix = sigma**2 * (1 + scaled_dist) * np.exp(-scaled_dist)
    
    return cov_matrix

def distance_matrix(beta):
    # Create coordinate grid for 8x8
    dim=beta.shape[0]
    dim_sqrt=int(np.sqrt(dim))
    coords = []
    for y in range(dim_sqrt):
        for x in range(dim_sqrt):
            coords.append((x, y))
    # Initialize distance matrix (64x64)
    dist_mat = np.zeros((dim, dim))
    
    # Calculate distances
    for i in range(dim):
        for j in range(dim):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            dist_mat[i, j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    return dist_mat


def sigma_distribution(beta,b,rho,sgm,n,a_rho,b_rho,a_sgm,b_sgm):
    # --- 1. Probability of prior
    roh_dist = stats.lognorm(s=b_rho, scale=np.exp(a_rho))
    sgm_dist = stats.lognorm(s=b_sgm, scale=np.exp(a_sgm))
    roh_prob=roh_dist.pdf(rho)
    sgm_prob=sgm_dist.pdf(sgm)

    # --- 2. Creation of Distance Matrix

    dist_mat=distance_matrix(beta)
    cov_mat=matern_covariance_32(dist_mat,rho,sgm)

    # --- 3. Likelihood 
    sum_of_squares = 0

    # Ensure beta is a column vector (z, 1) for calculations
    if beta.ndim == 1:
        beta_col = beta.reshape(-1, 1)
    else:
        beta_col = beta

    for i in range(n):
        b_i = b[i]

        # Ensure b_i is a column vector (z, 1)
        if b_i.ndim == 1:
            b_i_col = b_i.reshape(-1, 1)
        else:
            b_i_col = b_i

        deviation = b_i_col - beta_col
        sum_of_squares += deviation.T @ np.linalg.inv(cov_mat) @ deviation
    prob=(np.linalg.det(cov_mat))**(-n/2)*np.exp(-sum_of_squares/2)*roh_prob*sgm_prob
    return prob
    