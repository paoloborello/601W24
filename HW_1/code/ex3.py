import numpy as np


def pad_vector(vec: np.array, pos: list, val=0) -> np.array:
    v_padded = vec
    pos = sorted(pos)
    for p in pos:
        if p > len(vec):
            v_padded = np.append(v_padded, val)
        else:
            v_padded = np.insert(v_padded, p, val)
    return v_padded


def mra_euggm_ks(S: np.array, not_Edges: list, max_iter=1000, tol=1e-6) -> np.array:
    # initialize W and Theta
    W = S
    T = np.zeros_like(W)
    # get number of nodes (or variables)
    p = np.shape(S)[0]
    conv_flag = False
    for n_iter in range(max_iter):
        W_old = np.copy(W)
        for j in range(p):
            # mask to slice out appropriate rows and columns
            mask = np.ones(p, dtype=bool)
            # slice out row and column j
            mask[j] = False
            # slice out edges not incident to j
            for edge in not_Edges:
                if edge[0] == j:
                    mask[edge[1]] = False
                elif edge[1] == j:
                    mask[edge[0]] = False
            # slice out appropriate rows and columns
            W_star = W[mask, :][:, mask]
            s_star = S[mask, j]
            # solve for beta_star
            beta_star = np.linalg.solve(W_star, s_star)
            # list of nodes not incident to j (and coordinate j)
            not_j = np.arange(p)[np.invert(mask)]
            # pad beta_star with zeros at appropriate positions
            beta = pad_vector(beta_star, not_j, 0)
            # compute w (similar to w12 but element j is wrong)
            w = np.matmul(W, beta)
            # set element j to correct value
            w[j] = W[j, j]
            # set j-th row and column of W to w
            W[j] = w
            W[:, j] = w
            # if we converged, compute Theta
            if conv_flag:
                theta_22 = 1 / (S[j, j] - np.dot(w, beta))
                theta_12 = - beta * theta_22
                theta_12[j] = theta_22
                T[j] = theta_12
        # if converged break
        if conv_flag:
            break
        # check convergence
        norm = np.linalg.norm(W - W_old)
        if norm < tol:
            conv_flag = True
        print(f"Iter {n_iter + 1} - Norm: {round(norm, 7)}")
    return W, T


if __name__ == '__main__':
    # ESL 17.4 example

    # sample covariance matrix
    cov_S = np.array([[10, 1, 5, 4], [1, 10, 2, 6], [5, 2, 10, 3], [4, 6, 3, 10]], dtype=float)
    # list of missing edges
    not_E = [(0, 2), (1, 3)]
    Sigma, Theta = mra_euggm_ks(cov_S, not_E)
    print("W: ")
    print(np.round(Sigma, decimals=3))
    print("W_inv: ")
    print(np.round(Theta, decimals=3))
