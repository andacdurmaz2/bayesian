import numpy as np
from scipy.stats import invgamma, wishart
import scipy.special as sp

def sigma_e_given_rest(c_e, m, d_e, Y, X, b_mat):
     
    """
    Draw samples from sigma_e given the rest of the parameters
    
    Parameters:
    ce : int
        previous alpha
    de : int
        previous beta
    Y : int
        Y is result array -> dim: (mi x 1) => 25 x 1 
    X : array like matrix
        Basis function scaled by coeficients 1, evaluted at each year -> dim: (K x mi) => 8 x 25
    b_mat: array
        Coefficients for the Basisfunction -> dim: (K x 1) => 8 x 1
    """
    # update parameter c_e
    c_e_new = c_e + m / 2

    # compute the residual sum of squares
    residual_sum = 0.0
    for i in range(Y.shape[0]):
        r = Y[i] - (b_mat[i]@ X ).T
        residual_sum += r.T @ r

    d_e_new = d_e + 0.5 * residual_sum
    # Create inverse gamma distribution using scipy
    # Note: a is alpha in our class, scale is beta in our class
    sigma_e_new = invgamma(a=c_e_new, scale=d_e_new[0][0])

    return sigma_e_new, c_e_new, d_e_new[0][0]


def invwishart_rvs(df, scale, size=1):
    """
    Draw samples from inverse Wishart distribution
    
    Parameters:
    df : int
        Degrees of freedom
    scale : array_like
        Scale matrix (positive definite) -> dim: (mi x mi) => 25 x 25
    size : int
        Number of samples to draw
    """
    # Inverse Wishart: if X ~ Wishart(df, scale^{-1}), then X^{-1} ~ InvWishart(df, scale)
    scale_inv = np.linalg.inv(scale)
    
    if size == 1:
        # Draw one sample
        wishart_sample = wishart.rvs(df=df, scale=scale_inv)
        return np.linalg.inv(wishart_sample)
    else:
        # Draw multiple samples
        samples = []
        for _ in range(size):
            wishart_sample = wishart.rvs(df=df, scale=scale_inv)
            samples.append(np.linalg.inv(wishart_sample))
        return np.array(samples)

# Example usage:
df = 10
scale_matrix = np.array([[2, 0.5], [0.5, 1]])

# Draw samples from inverse Wishart
inv_wishart_samples = invwishart_rvs(df=df, scale=scale_matrix, size=1000)
print(f"Inverse Wishart samples shape: {inv_wishart_samples.shape}")
print('Hallo')

