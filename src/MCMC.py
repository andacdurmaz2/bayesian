import numpy as np
from src.conditionals import beta_draw, b_draw, sigma_b_draw, sigma_e_draw




def run_one_gibbs(y, X, sigma_b, sigma_e, b, beta, priors):
    """
    Runs one full iteration of the Gibbs sampler.

    Parameters
    ----------
    y : list
        List of n (m_i,) observation vectors.
    X : list
        List of n (m_i, z) design matrices.
    sigma_b : np.array
        Current state of the (z, z) random-effect covariance.
    sigma_e : float
        Current state of the residual variance.
    b : list
        List of n (z,) current random-effect vectors.
    beta : np.array
        Current state of the (z,) population-effect vector.
    priors : dict
        A dictionary containing all hyperparameters:
        - 'c_beta': Prior variance scalar for beta.
        - 'c_epsilon': Prior shape for sigma_e.
        - 'd_epsilon': Prior scale for sigma_e.
        - 'eta_b': Prior degrees of freedom for sigma_b.
        - 'S_b': Prior mean matrix (z, z) for sigma_b.

    Returns
    -------
    tuple
        A tuple of the updated parameters:
        (sigma_b_new, sigma_e_new, b_new, beta_new)
    """

    # Get constants from data
    n = len(y)
    m = sum(len(y_i) for y_i in y)

    # 1. Update population coefficients beta
    beta_new = beta_draw(
        sigma_b=sigma_b,
        sigma_e=sigma_e,
        y=y,
        c=priors['c_beta'],
        n=n,
        X=X
    )

    # 2. Update pixel-level random effects b_i
    #    Uses the NEWLY sampled beta
    b_new = b_draw(
        beta=beta_new,
        sigma_b=sigma_b,
        sigma_e=sigma_e,
        y=y,
        X=X
    )

    # 3. Update residual variance sigma_e
    #    Uses the NEWLY sampled b_new
    sigma_e_new = sigma_e_draw(
        beta=beta_new,  # (unused, but for signature)
        sigma_b=sigma_b, # (unused, but for signature)
        y=y,
        n=n,
        X=X,
        b=b_new,
        d=priors['d_epsilon'],
        c=priors['c_epsilon'],
        m=m
    )

    # 4. Update random-effect covariance sigma_b
    #    Uses the NEWLY sampled beta_new and b_new
    sigma_b_new = sigma_b_draw(
        beta=beta_new,
        b=b_new,
        y=y, # (unused, but for signature)
        n=n,
        etha_b=priors['eta_b'],
        S=priors['S_b']
    )

    # Return the new state
    return sigma_b_new, sigma_e_new, b_new, beta_new


def run_mcmc(y_list, X_list, priors, n_iter=2000, n_burn=1000):
    """
    Runs the full MCMC chain.
    """

    # Get dimensions
    n = len(y_list)
    z = X_list.shape[1]
    print(z)

    # --- Initialize the chain ---
    print("Initializing chain...")
    # Simple initial values
    sigma_e_curr = 1.0
    sigma_b_curr = np.eye(z)
    beta_curr = np.zeros(z)

    # Initialize b_i's to zero
    b_curr = [np.zeros(z) for _ in range(n)]

    # --- Storage for samples ---
    n_samples = n_iter - n_burn
    if n_samples <= 0:
        raise ValueError("n_iter must be greater than n_burn")

    beta_samples = np.zeros((n_samples, z))
    sigma_e_samples = np.zeros(n_samples)
    sigma_b_samples = np.zeros((n_samples, z, z))
    # We'll just store samples for the first group's b_i
    b_0_samples = np.zeros((n_samples, z))

    print(f"Running MCMC for {n_iter} iterations (burn-in: {n_burn})...")

    for i in range(n_iter):
        # Run one iteration
        sigma_b_curr, sigma_e_curr, b_curr, beta_curr = run_one_gibbs(
            y_list, X_list,
            sigma_b_curr, sigma_e_curr, b_curr, beta_curr,
            priors
        )

        # Store samples after burn-in
        if i >= n_burn:
            idx = i - n_burn
            beta_samples[idx, :] = beta_curr
            sigma_e_samples[idx] = sigma_e_curr
            sigma_b_samples[idx, :, :] = sigma_b_curr
            b_0_samples[idx, :] = b_curr[0]

        if (i + 1) % 500 == 0:
            print(f"Iteration {i+1}/{n_iter}...")

    print("MCMC finished.")

    return {
        'beta': beta_samples,
        'sigma_e': sigma_e_samples,
        'sigma_b': sigma_b_samples,
        'b_0': b_0_samples
    }