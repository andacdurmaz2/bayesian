import numpy as np
from scipy.stats import invgamma, invwishart, multivariate_normal


def update_beta(y_list, X_list, b_list, sigma2_eps, Sigma_b, c, n_pixels):
    """
    Update population-level spline coefficients beta.

    Parameters:
    -----------
    y_list : list of ndarrays
        Temperature observations for each pixel [y1, y2, ..., yn]
    X_list : list of ndarrays
        Design matrices for each pixel [X1, X2, ..., Xn]
    b_list : list of ndarrays
        Current pixel-specific random effects [b1, b2, ..., bn]
    sigma2_eps : float
        Current residual variance
    Sigma_b : ndarray
        Current covariance matrix of random effects
    c : float
        Prior variance parameter for beta
    n_pixels : int
        Number of pixels

    Returns:
    --------
    beta : ndarray
        Updated population coefficients
    """
    K = X_list[0].shape[1]  # Number of basis functions (8)
    Sigma_b_inv = np.linalg.inv(Sigma_b)

    # Compute Sigma_beta_conditional
    sum_term = np.zeros((K, K))
    for i in range(n_pixels):
        Xi = X_list[i]
        XtX = Xi.T @ Xi
        inner_inv = np.linalg.inv((1 / sigma2_eps) * XtX + Sigma_b_inv)
        sum_term += Sigma_b_inv @ inner_inv @ Sigma_b_inv

    Sigma_beta_cond = np.linalg.inv(
        n_pixels * Sigma_b_inv + (1 / c) * np.eye(K) - sum_term
    )

    # Compute mu_beta_conditional
    sum_mean = np.zeros(K)
    for i in range(n_pixels):
        Xi = X_list[i]
        yi = y_list[i]
        XtX = Xi.T @ Xi
        inner_inv = np.linalg.inv((1 / sigma2_eps) * XtX + Sigma_b_inv)
        sum_mean += Sigma_b_inv @ inner_inv @ ((1 / sigma2_eps) * Xi.T @ yi)

    mu_beta_cond = Sigma_beta_cond @ sum_mean

    # Sample from multivariate normal
    beta = multivariate_normal.rvs(mean=mu_beta_cond, cov=Sigma_beta_cond)

    return beta


def update_bi(y_list, X_list, i, beta, sigma2_eps, Sigma_b):
    """
    Update pixel-specific random effects bi for pixel i.

    Parameters:
    -----------
    y_list : list of ndarrays
        Temperature observations for all pixels
    X_list : list of ndarrays
        Design matrices for all pixels
    i : int
        Index of the pixel to update
    beta : ndarray
        Current population coefficients
    sigma2_eps : float
        Current residual variance
    Sigma_b : ndarray
        Current covariance matrix of random effects

    Returns:
    --------
    bi : ndarray
        Updated random effects for pixel i
    """
    # Extract data for pixel i
    yi = y_list[i]
    Xi = X_list[i]

    Sigma_b_inv = np.linalg.inv(Sigma_b)

    # Compute Sigma_bi_conditional
    Sigma_bi_cond = np.linalg.inv(
        (1 / sigma2_eps) * Xi.T @ Xi + Sigma_b_inv
    )

    # Compute mu_bi_conditional
    mu_bi_cond = Sigma_bi_cond @ (
            (1 / sigma2_eps) * Xi.T @ yi + Sigma_b_inv @ beta
    )

    # Sample from multivariate normal
    bi = multivariate_normal.rvs(mean=mu_bi_cond, cov=Sigma_bi_cond)

    return bi


def update_sigma2_eps(y_list, X_list, b_list, c_eps, d_eps, n_pixels):
    """
    Update residual variance sigma2_eps.

    Parameters:
    -----------
    y_list : list of ndarrays
        Temperature observations for each pixel [shape: (mi, 1) or (mi,)]
    X_list : list of ndarrays
        Design matrices for each pixel [shape: (mi, K)]
    b_list : list of ndarrays
        Current pixel-specific random effects [shape: (K,) or (K, 1)]
    c_eps : float
        Prior shape parameter
    d_eps : float
        Prior scale parameter
    n_pixels : int
        Number of pixels

    Returns:
    --------
    sigma2_eps : float
        Updated residual variance (single draw from inverse gamma)
    """
    # Compute total number of observations
    m = sum(len(yi) for yi in y_list)

    # Update shape parameter c_eps
    c_eps_new = c_eps + m / 2

    # Compute the sum of squared residuals
    sum_squared_residuals = 0.0
    for i in range(n_pixels):
        yi = y_list[i].reshape(-1, 1) if y_list[i].ndim == 1 else y_list[i]
        Xi = X_list[i]
        bi = b_list[i].reshape(-1, 1) if b_list[i].ndim == 1 else b_list[i]

        # Compute residual: r = y_i - X_i @ b_i
        r = yi - Xi @ bi
        # Add squared residuals: r^T @ r
        sum_squared_residuals += r.T @ r

    # Update scale parameter d_eps
    # Extract scalar if sum_squared_residuals is a 1x1 array
    if isinstance(sum_squared_residuals, np.ndarray):
        sum_squared_residuals = sum_squared_residuals.item()

    d_eps_new = d_eps + 0.5 * sum_squared_residuals

    # Sample from inverse gamma distribution
    # Note: a is shape parameter (alpha), scale is scale parameter (beta)
    sigma2_eps = invgamma.rvs(a=c_eps_new, scale=d_eps_new)

    return sigma2_eps


def update_Sigma_b(beta, b_list, eta_b, S, n_pixels):
    """
    Update random-effect covariance matrix Sigma_b.

    Parameters:
    -----------
    beta : ndarray
        Current population coefficients
    b_list : list of ndarrays
        Current pixel-specific random effects
    eta_b : float
        Prior degrees of freedom
    S : ndarray
        Prior scale matrix
    n_pixels : int
        Number of pixels

    Returns:
    --------
    Sigma_b : ndarray
        Updated covariance matrix
    """
    # Compute degrees of freedom
    eta_b_cond = eta_b + n_pixels

    # Compute sum of outer products
    sum_outer = np.zeros((len(beta), len(beta)))
    for i in range(n_pixels):
        bi = b_list[i]
        diff = bi - beta
        sum_outer += np.outer(diff, diff)

    # Compute scale matrix S_b_conditional
    S_b_cond = (1 / eta_b_cond) * (eta_b * S + sum_outer)

    # Sample from inverse Wishart
    # Note: scipy uses scale matrix, not "parameter" matrix
    Sigma_b = invwishart.rvs(df=eta_b_cond, scale=eta_b_cond * S_b_cond)

    return Sigma_b