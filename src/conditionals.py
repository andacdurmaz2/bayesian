import numpy as np
from scipy.stats import invgamma, invwishart



def b_draw(beta, sigma_b, sigma_e, y, X):

    """
    Draws a sample for each pixel-level random effect b_i.

    This function implements the full conditional distribution:
    b_i | beta, sigma_e, sigma_b, y_i ~ MVN(mu_bi, sigma_bi)

    This is based on the model:
    y_i ~ N(X_i @ b_i, sigma_e * I)
    b_i ~ N(beta, sigma_b)

    Parameters
    ----------
    beta : np.array
        The (z,) or (z, 1) population-level mean vector.
    sigma_b : np.array
        The (z, z) prior covariance matrix for b_i, (Sigma_b).
    sigma_e : float
        The scalar error variance (sigma^2_epsilon).
    y : list of np.array
        A list of n arrays. Each element y[i] is the (m_i, 1) or (m_i,)
        observation vector for group i.
    X : list of np.array
        A list of n arrays. Each element X[i] is the (m_i, z) design
        matrix for group i.

    Returns
    -------
    list
        A list of n numpy arrays, where each element is a (z,)
        draw for b_i.
    """

    # Get the number of groups
    n = len(y)

    # Get the dimension z from the shape of sigma_b
    z = sigma_b.shape[0]

    # --- Pre-compute Inverses and check dimensions ---
    # We add a small jitter for numerical stability
    jitter = 1e-6 * np.eye(z)

    try:
        sigma_b_inv = np.linalg.inv(sigma_b)
    except np.linalg.LinAlgError:
        sigma_b_inv = np.linalg.inv(sigma_b + jitter)

    # Ensure beta is a column vector (z, 1) for matrix math
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)

    # List to store the samples
    b_samples_list = []

    # --- Loop over all n groups ---
    for i in range(n):
        X_i = X
        y_i = y[i]

        # Ensure y_i is a column vector (m_i, 1)
        if y_i.ndim == 1:
            y_i = y_i.reshape(-1, 1)

        # --- 1. Calculate Posterior Covariance (Sigma_bi) ---

        # X_i' * X_i
        XtX = X_i.T @ X_i

        # Posterior precision: Lambda = (1/sigma_e)*X'X + sigma_b_inv
        lambda_bi = (1.0 / sigma_e) * XtX + sigma_b_inv

        # Posterior covariance: Sigma_bi = Lambda^{-1}
        try:
            sigma_bi = np.linalg.inv(lambda_bi)
        except np.linalg.LinAlgError:
            sigma_bi = np.linalg.inv(lambda_bi + jitter)

        # --- 2. Calculate Posterior Mean (mu_bi) ---

        # X_i' * y_i
        Xty = X_i.T @ y_i

        # K = (1/sigma_e)*X'y + sigma_b_inv @ beta
        K_i = (1.0 / sigma_e) * Xty + (sigma_b_inv @ beta)

        # mu_bi = Sigma_bi @ K
        mu_bi = sigma_bi @ K_i

        # --- 3. Draw the sample ---
        # np.random.multivariate_normal requires the mean to be a 1D array
        b_i_sample = np.random.multivariate_normal(mu_bi.flatten(), sigma_bi)

        b_samples_list.append(b_i_sample)

    return b_samples_list
    


def beta_draw(sigma_b, sigma_e, y, c, n, X):
    """
    Draws a single sample for the population coefficients beta.

    This function implements the full conditional distribution:
    beta | sigma_e, sigma_b, y ~ MVN(mu_beta, sigma_beta)
    Parameters
    ----------
    sigma_b : np.array
        The (z, z) prior covariance matrix for beta_i, (Sigma_b).
    sigma_e : float
        The scalar error variance (sigma^2_epsilon).
    y : list of np.array
        A list of n arrays. Each element y[i] is the (m_i, 1) observation
        vector for group i.
    c : float
        A scalar hyperparameter for the prior precision of beta.
    n : int
        The number of groups (the length of X and y).
    X : list of np.array
        A list of n arrays. Each element X[i] is the (m_i, z) design
        matrix for group i.

    Returns
    -------
    np.array
        A (z,) numpy array representing a single draw from the posterior.
    """

    # Get the dimension z from the shape of sigma_b
    z = sigma_b.shape[0]

    # --- Pre-compute Inverses and Identity ---
    # We add a small jitter for numerical stability in case
    # sigma_b is near-singular.
    jitter = 1e-6 * np.eye(z)

    try:
        sigma_b_inv = np.linalg.inv(sigma_b)
    except np.linalg.LinAlgError:
        sigma_b_inv = np.linalg.inv(sigma_b + jitter)

    I_z = np.eye(z)

    # --- Initialize Summation Terms ---

    # This is the large summation term in the precision matrix (Lambda)
    # sum( sigma_b_inv @ ( (1/sigma_e)*X_i'X_i + sigma_b_inv )^{-1} @ sigma_b_inv )
    sum_precision_term = np.zeros((z, z))

    # This is the large summation term for the mean (K = Lambda @ mu)
    # sum( sigma_b_inv @ ( (1/sigma_e)*X_i'X_i + sigma_b_inv )^{-1} @ (1/sigma_e)*X_i'y_i )
    sum_mean_term_K = np.zeros((z, 1))

    # --- Loop over all n groups ---
    for i in range(n):
        X_i = X
        y_i = y[i]

        # Ensure y_i is a column vector (m_i, 1)
        if y_i.ndim == 1:
            y_i = y_i.reshape(-1, 1)

        # --- Calculate intermediate terms ---
        # X_i' * X_i
        XtX = X_i.T @ X_i
        # X_i' * y_i
        Xty = X_i.T @ y_i

        # ( (1/sigma_e) * X_i'X_i + sigma_b_inv )
        inner_term = (1.0 / sigma_e) * XtX + sigma_b_inv

        # ( (1/sigma_e) * X_i'X_i + sigma_b_inv )^{-1}
        try:
            inner_inv = np.linalg.inv(inner_term)
        except np.linalg.LinAlgError:
            inner_inv = np.linalg.inv(inner_term + jitter)

        # --- Add to the Precision Sum ---
        # term_i = sigma_b_inv @ inner_inv @ sigma_b_inv
        sum_precision_term += sigma_b_inv @ inner_inv @ sigma_b_inv

        # --- Add to the Mean Sum (K) ---
        # term_i = sigma_b_inv @ inner_inv @ ( (1/sigma_e) * X_i'y_i )
        sum_mean_term_K += sigma_b_inv @ inner_inv @ ( (1.0 / sigma_e) * Xty )

    # --- 1. Calculate the Posterior Precision Matrix (Lambda) ---
    # Lambda = n*Sigma_b_inv + (1/c)*I - sum_precision_term
    lambda_beta = (n * sigma_b_inv) + ((1.0 / c) * I_z) - sum_precision_term

    # --- 2. Calculate the Posterior Covariance Matrix (Sigma) ---
    # Sigma = Lambda^{-1}
    try:
        sigma_beta = np.linalg.inv(lambda_beta)
    except np.linalg.LinAlgError:
        sigma_beta = np.linalg.inv(lambda_beta + jitter)

    # --- 3. Calculate the Posterior Mean (mu) ---
    # mu = Sigma @ K
    mu_beta = sigma_beta @ sum_mean_term_K

    # --- 4. Draw the sample from MVN(mu, Sigma) ---
    # np.random.multivariate_normal requires the mean to be a 1D array
    beta_sample = np.random.multivariate_normal(mu_beta.flatten(), sigma_beta)

    return beta_sample

def sigma_e_draw(beta, sigma_b, y, n, X, b, d, c, m):
    """
    Draws a single sample for the residual variance sigma_e^2.

    This function implements the full conditional distribution:
    sigma_e^2 | beta, b, Sigma_b, y ~ IG(c_e_posterior, d_e_posterior)

    Parameters
    ----------
    beta : np.array
        Population coefficients (unused in this step, but part of state).
    sigma_b : np.array
        Covariance matrix for b_i (unused in this step, but part of state).
    y : list of np.array
        A list of n arrays. Each element y[i] is the (m_i,) or (m_i, 1)
        observation vector for group i.
    n : int
        The number of groups (the length of X, y, and b).
    X : list of np.array
        A list of n arrays. Each element X[i] is the (m_i, z) design
        matrix for group i.
    b : list of np.array
        A list of n arrays. Each element b[i] is the (z,) or (z, 1)
        current coefficient vector for group i.
    d : float
        The prior scale parameter (d_epsilon) for the Inverse-Gamma.
    c : float
        The prior shape parameter (c_epsilon) for the Inverse-Gamma.
    m : int
        The total number of observations (sum of all m_i across all n groups).

    Returns
    -------
    float
        A single draw for sigma_e^2 from its posterior distribution.
    """

    # 1. Calculate posterior shape parameter
    # c_e_post = c_e + m/2
    c_posterior = c + m / 2.0

    # 2. Calculate the total Sum of Squared Errors (SSE)
    # SSE = sum_{i=1}^n (y_i - X_i*b_i)' * (y_i - X_i*b_i)
    total_sse = 0.0
    for i in range(n):
        y_i = y[i].flatten()  # Ensure 1D (m_i,)
        X_i = X            # (m_i, z)
        b_i = b[i].flatten()  # Ensure 1D (z,)

        # Calculate residuals: r_i = y_i - X_i @ b_i
        residuals = y_i - (X_i @ b_i)

        # Add squared errors for group i: r_i' @ r_i
        # Using np.dot for 1D arrays is equivalent to inner product
        total_sse += np.dot(residuals, residuals)

    # 3. Calculate posterior scale parameter
    # d_e_post = d_e + (1/2) * SSE
    d_posterior = d + 0.5 * total_sse

    # 4. Draw from the posterior Inverse-Gamma(c_posterior, d_posterior)
    #
    # We can sample from an Inverse-Gamma(a, b) (shape, scale)
    # by sampling from a Gamma(a, 1/b) (shape, rate)
    # and taking the reciprocal.
    # numpy.random.gamma(shape, scale) uses scale = 1/rate.

    # Draw a numeric sample from the Inverse-Gamma posterior.
    # Calling `invgamma(...)` returns a frozen distribution object; we need
    # to call `.rvs(...)` to obtain a numeric sample.
    sigma_e_squared_sample = invgamma.rvs(a=c_posterior, scale=d_posterior)

    return float(sigma_e_squared_sample)


def sigma_b_draw(beta, b, y, n, etha_b, S):
    """
    Draws a single sample for the random-effect covariance matrix Sigma_b.

    This function implements the full conditional distribution:
    Sigma_b | beta, b, y ~ IW(eta_posterior, S_posterior)

    It assumes the common conjugate prior IW(etha_b, S_0), where
    the prior scale matrix S_0 is parameterized as S_0 = etha_b * S.

    Parameters
    ----------
    beta : np.array
        The (z,) or (z, 1) population-level mean vector.
    b : list of np.array
        A list of n arrays. Each element b[i] is the (z,) or (z, 1)
        current coefficient vector for group i.
    y : list (unused)
        The observation data (conditionally independent given b).
    n : int
        The number of groups (the length of b).
    etha_b : float
        The prior degrees of freedom (eta_b).
    S : np.array
        The (z, z) prior *mean* matrix. The prior scale matrix
        is calculated as S_0 = etha_b * S.

    Returns
    -------
    np.array
        A (z, z) matrix drawn from the posterior Inverse-Wishart.
    """

    # Get the dimension z from the shape of beta
    z = beta.shape[0]

    # --- 1. Calculate Posterior Degrees of Freedom ---
    # eta_posterior = etha_b + n
    eta_posterior = etha_b + n

    # --- 2. Calculate Posterior Scale Matrix ---

    # Start with the prior scale matrix: S_0 = etha_b * S
    S_0 = etha_b * S

    # Ensure beta is a column vector (z, 1) for calculations
    if beta.ndim == 1:
        beta_col = beta.reshape(-1, 1)
    else:
        beta_col = beta

    # Calculate the sum of squared deviations
    sum_of_squares = np.zeros((z, z))

    for i in range(n):
        b_i = b[i]

        # Ensure b_i is a column vector (z, 1)
        if b_i.ndim == 1:
            b_i_col = b_i.reshape(-1, 1)
        else:
            b_i_col = b_i

        # Calculate deviation: (b_i - beta)
        deviation = b_i_col - beta_col

        # Calculate outer product: (b_i - beta)(b_i - beta)'
        # This is (z, 1) @ (1, z) -> (z, z)
        sum_of_squares += deviation @ deviation.T

    # The posterior scale matrix: S_posterior = S_0 + sum_of_squares
    S_posterior = 1 / eta_posterior * (S_0 + sum_of_squares)

    # --- 3. Draw from the posterior Inverse-Wishart ---

    # Add a small jitter for numerical stability, as the
    # scale matrix must be positive definite.
    jitter = 1e-6 * np.eye(z)

    # Use scipy.stats.invwishart(df, scale)
    try:
        sigma_b_sample = invwishart.rvs(df=eta_posterior, scale=S_posterior)
    except np.linalg.LinAlgError:
        # If it fails (e.g., not positive definite), add jitter
        sigma_b_sample = invwishart.rvs(df=eta_posterior, scale=S_posterior + jitter)

    return sigma_b_sample


