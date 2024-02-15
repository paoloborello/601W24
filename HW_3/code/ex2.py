import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal


def gen_data(mu, Lambda, n=100, scale_1=1, scale_2=np.sqrt(0.4)):
    # generate data from a factor analysis model
    # inputs: mu: q x 1 mean vector
    #         Lambda: q x p factor loading matrix
    #         n: number of samples
    # returns: Y: n x q data matrix
    # generate latent variables
    p = Lambda.shape[1]
    X = np.random.normal(size=(n, p), scale=scale_1)
    # generate error
    q = mu.shape[0]
    W = np.random.normal(size=(n, q), scale=scale_2)
    # generate data
    Y = np.tile(mu, (n, 1)) + X @ Lambda.T + W
    return Y


def log_likelihood_FA_mmodel(mu, Lambda, Psi, Y):
    # compute the log-likelihood of the data under the factor analysis model
    # inputs: mu: q x 1 mean vector
    #         Lambda: q x p factor loading matrix
    #         Psi: q x q error covariance matrix
    #         Y: n x q data matrix
    # returns: ll: log-likelihood
    # helper matrix
    mean = mu
    cov = Lambda @ Lambda.T + Psi
    # inv_cov = np.linalg.inv(cov)
    # inv_cov = np.tile(inv_cov, (Y.shape[0], 1, 1))
    # Y_norm = Y - np.tile(mean, (Y.shape[0], 1))
    # log-likelihood
    ll = multivariate_normal.logpdf(Y, mean=mean, cov=cov).sum()
    # ll = - (Y.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] +
    #         np.einsum('ij,ijk,ik->', Y_norm, inv_cov, Y_norm))
    return ll


def E_step_FA_mmodel(mu, Lambda, Psi, Y):
    # E-step of the EM algorithm for factor analysis
    # inputs: mu: q x 1 mean vector
    #         Lambda: q x p factor loading matrix
    #         Psi: q x q error covariance matrix
    #         Y: n X q data vector
    # returns: e_X: n x p latent variable means, each row is a mean vector
    #          v_X: n x p x p latent variable "covariance" tensor,
    #               each slice over the first axis is a matrix E[X_i X_i^T|Y_i]
    # helper matrix
    beta = Lambda.T @ np.linalg.inv(Lambda @ Lambda.T + Psi)
    # normalize Y by mu
    Y_norm = Y - np.tile(mu, (Y.shape[0], 1))
    # n x p matrix of latent variable means
    e_X = Y_norm @ beta.T
    # n x p x p tensor of outer products of rows of e_X
    outer_prod_tensor = np.einsum('ij,ik->ijk', e_X, e_X)
    # create p x p matrix common to all observations
    matrix = np.eye(Lambda.shape[1]) - beta @ Lambda
    # sum matrix with outer_prod_tensor to return n x p x p tensor
    v_X = np.tile(matrix, (Y.shape[0], 1, 1)) + outer_prod_tensor
    return e_X, v_X


def M_step_FA_mmodel(Y, e_X, v_X):
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
    # factor loading matrix
    Lambda = np.einsum('ij,ik->jk', Y_norm, e_X) @ np.linalg.inv(np.sum(v_X, axis=0))
    # error covariance matrix
    first_tens = np.einsum('ij,ik->ijk', Y_norm, Y_norm)
    second_tens = np.einsum('ij,ik->ijk',  e_X @ Lambda.T, Y_norm)
    Psi = (first_tens - second_tens).mean(axis=0)
    return mu, Lambda, Psi


if __name__ == '__main__':
    F = np.array([[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]], dtype=float).T
    m = np.zeros(7)
    data = gen_data(m, F, n=100)

    # part a
    nsim = 1000
    max_iter = 1000
    for sim in range(nsim):
        Lambda_hat = F + np.random.normal(scale=0.01, size=F.shape)
        Psi_hat = np.eye(F.shape[0])
        mu_hat = np.zeros(F.shape[0])

        for i in range(max_iter):
            cond_E_X, cond_V_X = E_step_FA_mmodel(mu_hat, Lambda_hat, Psi_hat, data)
            old_mu, old_Lambda, old_Psi = mu_hat, Lambda_hat, Psi_hat
            mu_hat, Lambda_hat, Psi_hat = M_step_FA_mmodel(data, cond_E_X, cond_V_X)
            if np.allclose(old_mu, mu_hat) and np.allclose(old_Lambda, Lambda_hat) and np.allclose(old_Psi, Psi_hat):
                log_l = log_likelihood_FA_mmodel(mu_hat, Lambda_hat, Psi_hat, data)
                print(f'Sim {sim+1} converged at iteration {i+1}, final log-likelihood: {np.round(log_l,3)}')
                break

    # part b
