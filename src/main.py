import numpy as np
from src.data_import import data
from src.BSpline import BSplineBasis
from src.MCMC import run_mcmc


if __name__ == "__main__":
    # --- 1. Create B-Spline Basis ---


    K=8 # number of basis functions. Calles "K" in Biostatistics paper
    spline_basis=BSplineBasis(t0=0,t1=30, n_basis=K,degree=3)
    ts, B= spline_basis.evaluate()
    print('B:',B.T.shape)
    print('y:',len(data))
    print('y[i]',data[0].shape)


    # --- 2. Import CSV data ---


    # --- 3. Define Priors ---
    priors = {
        'c_beta': 100.0,            # Vague prior for beta
        'c_epsilon': 0.01,          # Vague prior for sigma_e
        'd_epsilon': 0.01,          # Vague prior for sigma_e
        'eta_b': K + 2,    # Prior df for sigma_b (min for defined mean)
        'S_b': np.eye(K)   # Prior mean matrix for sigma_b
    }
 # --- 4. Run MCMC ---
    samples = run_mcmc(data, B.T, priors, n_iter=3000, n_burn=1500)
    # --- 5. Show Results ---
    print("\n--- Posterior Means vs. True Values ---")

    print("\nBeta (Population Coefficients):")
    print(f"  Posterior Mean: {np.mean(samples['beta'], axis=0)}")

    print("\nsigma_e (Residual Variance):")
    print(f"  Posterior Mean: {np.mean(samples['sigma_e']):.4f}")

    print("\nsigma_b (Random-Effect Covariance):")
    print(f"  Posterior Mean:\n{np.mean(samples['sigma_b'], axis=0)}")


    print("\nb_0 (Random Effects for Group 0):")
    print(f"  Posterior Mean: {np.mean(samples['b_0'], axis=0)}")
# --- 4. Show Results ---
    print("\n--- Posterior Means vs. True Values ---")

    print("\nBeta (Population Coefficients):")
    print(f"  Posterior Mean: {np.mean(samples['beta'], axis=0)}")


    print("\nsigma_e (Residual Variance):")
    print(f"  Posterior Mean: {np.mean(samples['sigma_e']):.4f}")


    print("\nsigma_b (Random-Effect Covariance):")
    print(f"  Posterior Mean:\n{np.mean(samples['sigma_b'], axis=0)}")
 

    print("\nb_0 (Random Effects for Group 0):")
    print(f"  Posterior Mean: {np.mean(samples['b_0'], axis=0)}")







