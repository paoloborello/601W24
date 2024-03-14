import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
import pandas as pd
np.random.seed(65987833)


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
    # log-likelihood
    ll = multivariate_normal.logpdf(Y, mean=mean, cov=cov)
    return ll.sum()


def E_step_FA_mmodel(mu, Lambda, Psi, Y):
    # E-step of the EM algorithm for factor analysis
    # inputs: mu: q x 1 mean vector
    #         Lambda: q x p factor loading matrix
    #         Psi: q x q error covariance matrix
    #         Y: n X q data vector
    # returns: e_X: n x p latent variable means, each row is a mean vector E[X_i|Y_i]
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
    Psi = np.diag(np.diag((first_tens - second_tens).mean(axis=0)))
    return mu, Lambda, Psi


def EM_FA_mmodel(Y, Lambda_start, max_iters=1000, tol=1e-6):
    # EM algorithm for factor analysis
    # Y: n x q data matrix
    # p: number of factors
    # max_iter: maximum number of iterations
    # tol: convergence tolerance
    # returns: mu: q x 1 mean vector
    #          Lambda: q x p factor loading matrix
    #          Psi: q x q error covariance matrix
    #          log_l: log-likelihood
    n, q = Y.shape
    # initialize mu, Lambda, Psi
    mu = Y.mean(axis=0)
    Lambda = Lambda_start
    Psi = np.eye(q)
    # EM algorithm
    for _ in range(max_iters):
        # E-step
        e_X, v_X = E_step_FA_mmodel(mu, Lambda, Psi, Y)
        # M-step
        mu_new, Lambda_new, Psi_new = M_step_FA_mmodel(Y, e_X, v_X)
        # check convergence
        if np.allclose(Lambda, Lambda_new, atol=tol) and np.allclose(Psi, Psi_new, atol=tol):
            break
        mu, Lambda, Psi = mu_new, Lambda_new, Psi_new
    # log-likelihood
    log_like = log_likelihood_FA_mmodel(mu, Lambda, Psi, Y)
    return mu, Lambda, Psi, log_like


def projection_norm(A, B):
    # compute the Frobenius norm of difference between two projection matrices P_A and P_B
    # inputs: A: q x p matrix
    #         A: q x p matrix
    # returns: ||A(A^t A)^{-1} A^t - B(B^t B)^{-1} B^t||_F
    P_A = A @ np.linalg.inv(A.T @ A) @ A.T
    P_B = B @ np.linalg.inv(B.T @ B) @ B.T
    return np.linalg.norm(P_A - P_B, ord='fro')


if __name__ == '__main__':
    F = np.array([[1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1]],
                 dtype=float).T
    m = np.zeros(7)
    data = gen_data(m, F, n=100)

    # part a
    nsim = 500
    max_iter = 1000
    sigmas = [0.001, 0.01, 0.1, 1, 10]
    log_ls = np.zeros((len(sigmas), nsim))
    Lambda_estimates = []
    for i, sigma in enumerate(tqdm(sigmas)):
        lambdas_sigma_runs = []
        for sim in range(nsim):
            Lambda0 = F + np.random.normal(scale=sigma, size=F.shape)
            _, Lambda_star, _, log_l_star = EM_FA_mmodel(data, Lambda0, max_iter)
            log_ls[i, sim] = log_l_star
            lambdas_sigma_runs.append(Lambda_star)
        Lambda_estimates.append(lambdas_sigma_runs[np.argmax(log_ls[i, :])])

    print(f"True log-likelihood: {np.round(log_likelihood_FA_mmodel(m, F, 0.4 * np.eye(m.shape[0]), data), 3)}"
          f"\n\n"
          f"Estimated maximal log-likelihoods: \n"
          f"""{pd.DataFrame({"sigma": sigmas, "log_l": log_ls.max(axis=1).round(3)})}""")
    for i, est in enumerate(Lambda_estimates):
        print(f"""{"-" * 70}""")
        print(f"sigma: {sigmas[i]}")
        print(f"Best Estimate:\n"
              f"{est}")

    # part b
    nsim = 100
    norms = np.zeros((len(sigmas), nsim))
    Lambda_estimates = []
    Psi_estimates = []
    log_ls = np.zeros((len(sigmas), nsim))
    for i, sigma in enumerate(tqdm(sigmas)):
        lambdas_sigma_runs = []
        psis_sigma_runs = []
        for sim in range(nsim):
            Lambda_0 = np.random.normal(scale=sigma, size=F.shape)
            _, Lambda_star, _, log_l_star = EM_FA_mmodel(data, Lambda_0, max_iter)
            norms[i, sim] = projection_norm(F, Lambda_star)
            log_ls[i, sim] = log_l_star
            lambdas_sigma_runs.append(Lambda_star)
        best_index = np.argmin(norms[i, :])
        Lambda_estimates.append(lambdas_sigma_runs[best_index])

    for i in range(len(sigmas)):
        print(f"""{"-" * 70}\n"""
              f"sigma: {sigmas[i]}\n"
              f"Minimal distance in column space: {round(norms[i].min(), 3)}\n\n"
              f"Estimate achieving minimal distance in column space for sigma = {sigmas[i]}:\n "
              f"{np.round(Lambda_estimates[i], 3)}\n")
        print(f"Log-likelihood of our estimate: "
              f"{round(log_ls[i, np.argmin(norms[i, :])], 3)}")
        print(f"Log-likelihood of the true model: "
              f"{round(log_likelihood_FA_mmodel(m, F, 0.4 * np.eye(m.shape[0]), data), 3)}")
    print(f"""{"-" * 70}\n""")
    print(f"True factor matrix: \n {F}")


