import numpy as np
from scipy.stats import invgamma
import scipy.special as sp

def sigma_e_given_rest(c_e, m, d_e, Y, X, b_mat):
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

