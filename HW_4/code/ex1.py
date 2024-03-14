import numpy as np
from HW_3.code.ex2 import EM_FA_mmodel


def norm_vec(x):
    # x: input vector
    # return: normalized vector
    return x / np.linalg.norm(x)


def gen_data(L, n=200):
    # generate data from factor model
    # Lambda: factor loading matrix
    # n: number of samples
    q = L.shape[1]
    X = np.random.normal(size=(n, q))
    return X @ L.T + np.random.normal(size=(n, L.shape[0]))


def PCA(X):
    # X: data matrix
    # return: principal components
    # centering
    X = X - np.mean(X, axis=0)
    # svd
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return U @ np.diag(S)


if __name__ == '__main__':
    # generate data
    Lambda = np.array([[1, 0, 0],
                       [1, 0.001, 0],
                       [0, 0, 10]]
                      )
    Y = gen_data(Lambda, n=1000)
    # PCA
    pc_comp = PCA(Y)
    # EM
    _, Lambda, Psi, _ = EM_FA_mmodel(Y, Lambda_start=np.random.normal(size=(3, 3)))
    print(np.dot(norm_vec(pc_comp[:, 0]), norm_vec(Y[:, 2])))
    print(Lambda)
    print(Psi)
