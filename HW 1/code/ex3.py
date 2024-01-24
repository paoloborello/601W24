import numpy as np


def pad_vector(vec: np.array, pos: list, val=0) -> np.array:
    v_padded = vec
    for p in pos:
        if p > len(vec):
            v_padded = np.append(v_padded, val)
        else:
            v_padded = np.insert(v_padded, p, val)
    return v_padded


def mra_euggm_ks(S: np.array, not_E: list, max_iter=1000, tol=1e-6) -> np.array:
    # initialize W
    W = S
    # get number of nodes
    p = np.shape(S)[0]
    for n_iter in range(max_iter):
        W_old = np.copy(W)
        for j in range(p):
            # mask to slice out appropriate rows and columns
            mask = np.ones(p, dtype=bool)
            # slice out row and column j
            mask[j] = False
            W_11 = W[mask, :][:, mask]
            # slice out edges not incident to j
            for edge in not_E:
                if edge[0] == j:
                    mask[edge[1]] = False
                elif edge[1] == j:
                    mask[edge[0]] = False
            # slice out appropriate rows and columns
            W_star = W[mask, :][:, mask]
            s_star = S[mask, j]
            # solve for beta_star
            beta_star = np.linalg.solve(W_star, s_star)
            # correct up to this point

            # list of nodes not incident to j
            # mask[j] = True
            not_j = np.arange(p)[np.invert(mask)]
            # pad beta_star with zeros at appropriate positions
            beta = pad_vector(beta_star, not_j, 0)
            print(f"""{"-" * 40}""")
            print(j, not_j)
            print(beta)
            w = np.matmul(W, beta)
            w[j] = W[j, j]

            # w_12 = np.matmul(W_11, beta)
            # w_12 = pad_vector(w_12, [j], W[j, j])
            print(f"""{"__" * 40}""")
            print(f"""w: {w}""")
            print(f"""W before up: {W}""")
            W[j] = w
            W[:, j] = w
            print(f"""W after up: {W}""")


        # check convergence
        if np.linalg.norm(W - W_old) < tol:
            break

    print(f"""final iter: {n_iter}""")
    return W


def main():
    # ESL 17.4 example
    # sample covariance matrix
    cov_S = np.array([[10, 1, 5, 4], [1, 10, 2, 6], [5, 2, 10, 3], [4, 6, 3, 10]], dtype=float)
    # list of missing edges
    not_E = [(0, 2), (1, 3)]
    A = mra_euggm_ks(cov_S, not_E)


if __name__ == '__main__':
    main()
