import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm


def hotellingT2(sample: np.array) -> float:
    # function to compute the Hotelling T2 statistic
    # input: sample matrix X (n x p)
    # output: Hotelling T2 statistic

    # compute sample mean and covariance
    sample_mean = np.mean(sample, axis=0)
    sample_cov = np.cov(sample, rowvar=False)
    # compute T2 statistic
    N = sample.shape[0]
    return N * np.matmul(np.matmul(sample_mean, np.linalg.inv(sample_cov)), sample_mean)


def random_psd_matrix(p: int, scale=1) -> np.array:
    # function to generate a random symmetric psd matrix
    # input: dimension p
    # output: random symmetric psd matrix

    # generate a random pxp matrix,
    # then create a symmetric matrix by multiplying it with its transpose
    helper = np.random.normal(size=(p, p), scale=scale)
    return np.dot(helper.T, helper)


def bootstrapT(sample: np.array) -> float:
    sample_mean = np.mean(sample, axis=0)
    return np.sum(np.abs(sample_mean))


if __name__ == '__main__':
    # set seed
    np.random.seed(65987833)

    # part (c) and (d)
    # for several dimensions p
    for p in (3, 10, 40, 80):
        # set sample size
        n = 100
        # mean set to be a vector of zeros to test under the null hypothesis
        mu = np.zeros(p)
        # covariance matrix is a random symmetric psd matrix with independent entries
        # generate a random pxp matrix,
        # then create a symmetric matrix by multiplying it with its transpose
        sigma = random_psd_matrix(p)

        # generate nsim samples from multivariate normal distribution
        # and compute the Hotelling T2 statistic for each sample
        nsim = 1000
        t2s = np.zeros(nsim)
        for sim in range(nsim):
            samples = np.random.multivariate_normal(mean=mu, cov=sigma, size=(n,))
            t2s[sim] = hotellingT2(samples)

        # compute type I error estimate for several significance levels
        alphas = np.linspace(0, 1, 100)
        coverages = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            coverages[i] = np.mean(t2s > scipy.stats.chi2.ppf(1 - alpha, df=p))

        # plot significance level vs type I error estimate
        plt.plot(alphas, coverages)
        # plot the line y=x for reference
        plt.plot((0, 1), (0, 1), linestyle='--')
        plt.gca().set_aspect('equal')
        plt.title(f"$p={p}$")
        plt.xlabel(r"significance level $\alpha$")
        plt.ylabel(r"type I error estimate")
        # save figure and clear plot
        plt.savefig(f"../images/ex4_{p}.png")
        plt.show()

    # part (f)
    n = 100
    p = 120
    mu = np.zeros(p)
    sigma = random_psd_matrix(p)

    nsim = 100
    bootstrap_samples = 1000
    ts = np.zeros(shape=(nsim, ), dtype=float)
    bootstrap_ts = np.zeros(shape=(nsim, bootstrap_samples), dtype=float)
    for sim in tqdm(range(nsim)):
        samples = np.random.multivariate_normal(mean=mu, cov=sigma, size=(n,))
        S_n = np.cov(samples, rowvar=False)
        ts[sim] = bootstrapT(samples)
        for b in range(bootstrap_samples):
            samples = np.random.multivariate_normal(mean=mu, cov=S_n, size=(n,))
            t = bootstrapT(samples)
            bootstrap_ts[sim, b] = t

    # compute type I error estimate for several significance levels
    alphas = np.linspace(0, 1, 100)
    coverages = np.zeros_like(alphas)
    for i, alpha in enumerate(alphas):
        qs = np.quantile(bootstrap_ts, 1 - alpha, axis=1)
        coverages[i] = np.mean(ts > qs)

    # plot significance level vs type I error estimate
    plt.plot(alphas, coverages)
    # plot the line y=x for reference
    plt.plot((0, 1), (0, 1), linestyle='--')
    plt.gca().set_aspect('equal')
    plt.title(f"$p={p}$")
    plt.xlabel(r"significance level $\alpha$")
    plt.ylabel(r"type I error estimate")
    # save figure and clear plot
    plt.savefig(f"../images/ex4_120.png")
    plt.show()
