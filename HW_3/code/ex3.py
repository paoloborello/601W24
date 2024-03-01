import numpy as np

from ex2 import *
import sklearn.linear_model as lm


def M_step_FA_mmodel_LASSO(Y, e_X, v_X, Lambda_k):
    # M-step of the EM algorithm for factor analysis
    # Y: n x q data matrix
    # e_X: n x p latent variable means
    # v_X: n x p x p latent variable "covariance" tensor
    # returns: mu: q x 1 mean vector
    #          Lambda: q x p factor loading matrix
    #          Psi: q x q error covariance matrix
    # mean vector
    mu = Y.mean(axis=0)
    # normalize Y by mu
    Y_norm = Y - np.tile(mu, (Y.shape[0], 1))
    # update Psi
    first_tens = np.einsum('ij,ik->ijk', Y_norm, Y_norm)
    second_tens = np.einsum('ij,ik->ijk', e_X @ Lambda_k.T, Y_norm)
    Psi = np.diag(np.diag((first_tens - second_tens).mean(axis=0)))
    # create E[X^t X|Y] p x p matrix
    E_XXT = np.sum(v_X, axis=0)
    # compute the Cholesky decomposition of E[XX^T], and its inverse, both p x p matrices
    chol_E_XXT = np.linalg.cholesky(E_XXT).T
    chol_inv_E_XXT = np.linalg.inv(chol_E_XXT)
    # create surrogate response matrix p x q and covariate matrix p x p
    Y_sur = chol_inv_E_XXT @ e_X.T @ Y_norm
    X_sur = chol_E_XXT
    # compute the LASSO estimate of Lambda
    Lambda = np.zeros(Lambda_k.shape)
    lasso = lm.Lasso(alpha=1, fit_intercept=False)
    for q in range(Y.shape[1]):
        # update row q of Lambda
        lam = 1
        penalty = lam * 2 * Y.shape[0] * Psi[q, q]
        lasso.set_params(alpha=penalty)
        Lambda[q] = lasso.fit(X_sur, Y_sur[:, q]).coef_
    # Lambda = np.einsum('ij,ik->jk', Y_norm, e_X) @ np.linalg.inv(np.sum(v_X, axis=0))
    # error covariance matrix
    return mu, Lambda, Psi


if __name__ == "__main__":
    print("Running M_step_FA_mmodel_LASSO")