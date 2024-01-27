import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # set seed
    np.random.seed(65987833)
    # for several dimensions p
    for p in (3, 10, 40, 80):
        # set sample size
        n = 100
        # mean set to be a vector of zeros to test under the null hypothesis
        mu = np.zeros(p)
        # covariance matrix is a random symmetric psd matrix with independent entries
        # generate a random pxp matrix,
        # then create a symmetric matrix by multiplying it with its transpose
        helper_matrix = np.random.normal(size=(p, p))
        sigma = np.dot(helper_matrix.T, helper_matrix)

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

