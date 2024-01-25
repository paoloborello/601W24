import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def hotellingT2(sample: np.array) -> float:
    sample_mean = np.mean(sample, axis=0)
    sample_cov = np.cov(sample, rowvar=False)
    N = sample.shape[0]
    return N * np.matmul(np.matmul(sample_mean, np.linalg.inv(sample_cov)), sample_mean)


if __name__ == '__main__':
    np.random.seed(65987833)
    for p in (3, 10, 40, 80):
        n = 100
        mu = np.zeros(p)
        helper_matrix = np.random.normal(size=(p, p))
        sigma = np.dot(helper_matrix.T, helper_matrix)

        nsim = 1000
        t2s = np.zeros(nsim)
        for sim in range(nsim):
            samples = np.random.multivariate_normal(mean=mu, cov=sigma, size=(n,))
            t2s[sim] = hotellingT2(samples)

        alphas = np.linspace(0, 1, 100)
        coverages = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            coverages[i] = np.mean(t2s > scipy.stats.chi2.ppf(1 - alpha, df=p))

        plt.plot(alphas, coverages)
        plt.plot((0, 1), (0, 1), linestyle='--')
        plt.gca().set_aspect('equal')
        plt.title(f"$p={p}$")
        plt.xlabel(r"significance level $\alpha$")
        plt.ylabel(r"type I error estimate")
        plt.savefig(f"../images/ex4_{p}.png")
        plt.show()

