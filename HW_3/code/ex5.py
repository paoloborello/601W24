import numpy as np
import pyreadr
import matplotlib.pyplot as plt


def linear_kernel(Y):
    # compute the linear kernel matrix
    # inputs: Y: n x p data matrix
    # returns: K: n x n kernel matrix
    return Y @ Y.T


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
    nyt_frame = pyreadr.read_r('./data/nytimes.RData')["nyt.frame"]
    labels = nyt_frame["class.labels"]
    labels = (labels == "art").values.astype(int)
    colors = np.array(["yellow", "blue"])[labels]
    data = nyt_frame.drop("class.labels", axis=1).to_numpy()

    kernels = [linear_kernel(data), rbf_kernel(data, gamma=10), poly_kernel(data)]
    kernel_names = ["Linear", "RBF", "Polynomial"]
    for i, kernel in enumerate(kernels):
        proj_1d = kernel_PCA_proj(kernel, 1)
        plt.scatter(proj_1d, np.zeros_like(proj_1d), c=colors, alpha=0.1)
        plt.title(f"1D {kernel_names[i]} PCA")
        plt.show()
        proj_2d = kernel_PCA_proj(kernel, 2)
        plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors)
        plt.show()
        proj_3d = kernel_PCA_proj(kernel, 3)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2], c=colors)
        plt.show()
