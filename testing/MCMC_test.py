import numpy as np
from src.MCMC import run_mcmc


if __name__ == "__main__":

    # --- 1. Simulate Data ---
    np.random.seed(42)

    n_groups = 50
    m_obs_per_group = 30
    z_features = 3

    # True parameters
    true_beta = np.array([5.0, -2.0, 1.0])
    true_sigma_b = np.array([
        [1.0, 0.5, 0.0],
        [0.5, 1.2, 0.3],
        [0.0, 0.3, 0.8]
    ])
    true_sigma_e = 1.5

    X_list = []
    y_list = []
    true_b_list = []

    print("Simulating data...")
    for _ in range(n_groups):
        # Draw group-level params
        b_i = np.random.multivariate_normal(true_beta, true_sigma_b)
        true_b_list.append(b_i)

        # Create design matrix X_i
        X_i = np.ones((m_obs_per_group, z_features))
        X_i[:, 1:] = np.random.randn(m_obs_per_group, z_features - 1)

        # Create observations y_i
        error = np.random.normal(0, np.sqrt(true_sigma_e), m_obs_per_group)
        y_i = X_i @ b_i + error

        X_list.append(X_i)
        y_list.append(y_i)

    print(f"Data simulated: n={n_groups}, m_i={m_obs_per_group}, z={z_features}")

    # --- 2. Define Priors ---
    priors = {
        'c_beta': 100.0,            # Vague prior for beta
        'c_epsilon': 0.01,          # Vague prior for sigma_e
        'd_epsilon': 0.01,          # Vague prior for sigma_e
        'eta_b': z_features + 2,    # Prior df for sigma_b (min for defined mean)
        'S_b': np.eye(z_features)   # Prior mean matrix for sigma_b
    }

    # --- 3. Run MCMC ---
    samples = run_mcmc(y_list, X_list, priors, n_iter=3000, n_burn=1500)

    # --- 4. Show Results ---
    print("\n--- Posterior Means vs. True Values ---")

    print("\nBeta (Population Coefficients):")
    print(f"  Posterior Mean: {np.mean(samples['beta'], axis=0)}")
    print(f"  True Value:     {true_beta}")

    print("\nsigma_e (Residual Variance):")
    print(f"  Posterior Mean: {np.mean(samples['sigma_e']):.4f}")
    print(f"  True Value:     {true_sigma_e:.4f}")

    print("\nsigma_b (Random-Effect Covariance):")
    print(f"  Posterior Mean:\n{np.mean(samples['sigma_b'], axis=0)}")
    print(f"  True Value:\n{true_sigma_b}")

    print("\nb_0 (Random Effects for Group 0):")
    print(f"  Posterior Mean: {np.mean(samples['b_0'], axis=0)}")
    print(f"  True Value:     {true_b_list[0]}")