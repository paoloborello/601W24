import numpy as np


def MDS(D, p=2):
    # compute the MDS projection
    # inputs: D: n x n distance matrix
    #         p: number of dimensions
    # returns: X: n x p data matrix
    # number of samples
    n = D.shape[0]
    # centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    # compute the kernel matrix
    K = - 0.5 * H @ D @ H
    # compute the eigenvalues and eigenvectors
    D, U = np.linalg.eig(K)
    # sort the eigenvalues and eigenvectors
    idx = np.argsort(D)[::-1]
    D = D[idx]
    U = U[:, idx]
    # get the number of non-zero eigenvalues and choose at most that many pcs
    p = min(p, np.sum(D > 0))
    # select the first p eigenvalues and eigenvectors
    D = D[:p]
    U = U[:, :p]
    # compute the MDS projection
    X = U @ np.diag(np.sqrt(D))
    return X


if __name__ == '__main__':
    D_matrix = np.array([[0, 587, 1212, 701, 1936, 604, 748, 2139, 2182, 543],
                         [587, 0, 920, 940, 1745, 1188, 713, 1858, 1737, 597],
                         [1212, 920, 0, 879, 831, 1726, 1631, 949, 1021, 1494],
                         [701, 940, 879, 0, 1374, 968, 1420, 1645, 1891, 1220],
                         [1936, 1745, 831, 1374, 0, 2339, 2451, 347, 959, 2300],
                         [604, 1188, 1726, 968, 2339, 0, 1092, 2594, 2734, 923],
                         [748, 713, 1631, 1420, 2451, 1092, 0, 2571, 2408, 205],
                         [2139, 1858, 949, 1645, 347, 2594, 2571, 0, 678, 2442],
                         [2182, 1737, 1021, 1891, 959, 2734, 2408, 678, 0, 2329],
                         [543, 597, 1494, 1220, 2300, 923, 205, 2442, 2329, 0]],
                        dtype=float)
    print(MDS(D_matrix, 2))
