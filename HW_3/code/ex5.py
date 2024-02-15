import numpy as np


def rbf_kernel(Y, gamma=None):
    # compute the RBF kernel matrix
    # inputs: Y: n x p data matrix
    #         gamma: kernel width
    # returns: K: n x n kernel matrix
    if gamma is None:
        gamma = 1 / Y.shape[1]
    n = Y.shape[0]
    Y_sq = np.sum(Y ** 2, axis=1)
    Y_sq = np.tile(Y_sq, (n, 1))
    K = Y_sq + Y_sq.T - 2 * Y @ Y.T
    return np.exp(- gamma * K)


def poly_kernel(Y, degree=3, gamma=None, c=0):
    # compute the polynomial kernel matrix
    # inputs: Y: n x p data matrix
    #         degree: degree of the polynomial
    #         gamma: scaling factor
    #         c: constant term
    # returns: K: n x n kernel matrix
    # compute the kernel matrix
    if gamma is None:
        gamma = 1 / Y.shape[1]
    K = (gamma * (Y @ Y.T) + c) ** degree
    return K


def kernel_PCA_proj(K, pcs=1):
    D, U = np.linalg.eig(K)
    # get the number of non-zero eigenvalues and choose at most that many pcs
    pcs = min(pcs, np.sum(D > 0))
    # sort the eigenvalues and eigenvectors
    idx = np.argsort(D)[::-1]
    D = D[idx]
    U = U[:, idx]
    return U[:, :pcs] @ np.diag(np.sqrt(D[:pcs]))


if __name__ == '__main__':
    data = np.array([[1, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 0, 0, 0]],
                    dtype=float).T
    kernel = rbf_kernel(data)
    print(kernel_PCA_proj(kernel, 1))
