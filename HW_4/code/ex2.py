import numpy as np
import pandas as pd
import torch
from HW_3.code.ex5 import linear_kernel, rbf_kernel, poly_kernel
from typing import TypeVar

Tensor = TypeVar('torch.tensor')
Array = TypeVar('numpy.ndarray')

def center_columns(X: Array) -> Array:
    # X: feature matrix
    # return: centered data matrix
    return X - np.mean(X, axis=0)


def log_transform(X: Array) -> Array:
    # X: feature matrix
    # return: log transformed data matrix
    return np.log(X + 1)


def discrete_transform(X: Array, center=True) -> Array:
    # X: feature matrix
    # return: discrete transformed data matrix
    if center:
        X = center_columns(X)
    return (X > 0).astype(float)


class PenalizedLogisticReg:
    def __init__(self) -> None:
        self.theta = None
        self.lamb = None

    @staticmethod
    def log_likelihood(X: Tensor, y: Tensor, theta: Tensor) -> Tensor:
        # X: feature matrix
        # y: labels
        # theta: parameters
        # return: log likelihood
        return torch.sum(y * (X @ theta) - torch.log(1 + torch.exp(X @ theta)))

    def loss(self, X: Tensor, y: Tensor, theta: Tensor, lamb: float) -> Tensor:
        # X: feature matrix
        # y: labels
        # theta: parameters
        # return: loss = - log_likelihood + L_2_penalty
        return - self.log_likelihood(X, y, theta) + lamb * torch.sum(theta ** 2)

    def train(self, X: Array, y: Array, lamb: float) -> None:
        # X: feature matrix
        # y: labels
        # lamb: penalty parameter
        # fit penalized logistic regression using gradient descent
        # convert data to tensors
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        # initialize parameters
        theta = torch.zeros(X.shape[1], requires_grad=True)
        # train model
        optimizer = torch.optim.Adam([theta], lr=0.001)
        for _ in range(1000):
            optimizer.zero_grad()
            loss = self.loss(X, y, theta, lamb)
            loss.backward()
            optimizer.step()
        # save parameters
        self.theta = theta.detach().numpy()

    def predict(self, X: Array) -> Array:
        # X: feature matrix
        # return: predicted labels
        return (1 + np.exp(-X @ self.theta) < 2).astype(int)

    def accuracy(self, X: Array, y: Array) -> float:
        # X: feature matrix
        # y: labels
        # return: accuracy
        return np.mean(self.predict(X) == y)

    def cv_train(self, X: Array, y: Array, lambdas: [float], k: int) -> None:
        # X: feature matrix
        # y: labels
        # lamb: penalty parameter
        # k: number of folds
        # fit penalized logistic regression using cross-validation
        CV_accuracies = []
        for lamb in lambdas:
            test_accuracy = 0
            # split data into k folds
            n = len(y)
            fold_size = n // k
            for i in range(k):
                # split data into train and test sets
                test_indices = list(range(i * fold_size, (i + 1) * fold_size))
                train_indices = list(set(range(n)) - set(test_indices))
                X_tr, y_tr = X[train_indices], y[train_indices]
                X_te, y_te = X[test_indices], y[test_indices]
                # train model
                self.train(X_tr, y_tr, lamb)
                # calculate accuracy
                test_accuracy += self.accuracy(X_te, y_te)
            # calculate average accuracy
            CV_accuracies.append(test_accuracy / k)
        # select lambda with highest accuracy
        self.lamb = lambdas[np.argmax(CV_accuracies)]
        # train model using selected lambda
        self.train(X, y, self.lamb)

    def __str__(self) -> str:
        return (f'PenalizedLogisticReg(\n'
                f'theta={self.theta})\n'
                f'lamb={self.lamb}')


class LDAClassifier:
    def __init__(self) -> None:
        self.pi_1 = None
        self.pi_0 = None
        self.mu_1 = None
        self.mu_0 = None
        self.sigma = None

    @staticmethod
    def discriminant_function(X: Array, mu: Array, sigma: Array, pi: float) -> Array:
        # X: feature matrix
        # mu: mean
        # sigma: covariance matrix
        # pi: class proportion
        # return: discriminant function
        sigma_inv = np.linalg.inv(sigma)
        return X @ sigma_inv @ mu - 0.5 * mu.T @ sigma_inv @ mu + np.log(pi)

    def train(self, X: Array, y: Array) -> None:
        # X: feature matrix
        # y: labels
        # fit LDA model
        # calculate class proportions
        n, p = X.shape
        # calculate class proportions
        self.pi_1 = np.mean(y == 1)
        self.pi_0 = 1 - self.pi_1
        # calculate the mean for each class
        self.mu_1 = np.mean(X[y == 1], axis=0)
        self.mu_0 = np.mean(X[y == 0], axis=0)
        # calculate the within-class scatter matrix
        sigma_1 = (X[y == 1] - self.mu_1).T @ (X[y == 1] - self.mu_1)
        sigma_0 = (X[y == 0] - self.mu_0).T @ (X[y == 0] - self.mu_0)
        # calculate the pooled within-class scatter matrix
        self.sigma = (sigma_0 + sigma_1) / (n - 1)

    def predict(self, X: Array) -> Array:
        # X: data matrix
        # return: predicted labels
        return (LDAClassifier.discriminant_function(X, self.mu_1, self.sigma, self.pi_1) >
                LDAClassifier.discriminant_function(X, self.mu_0, self.sigma, self.pi_0)).astype(int)

    def accuracy(self, X: Array, y: Array) -> float:
        # X: data matrix
        # y: labels
        # return: accuracy
        return np.mean(self.predict(X) == y).round(5)

    def __str__(self) -> str:
        return (f'LDAClassifier(\n'
                f'pi_1={self.pi_1},\n'
                f'pi_0={self.pi_0},\n'
                f'mu_1={self.mu_1},\n'
                f'mu_0={self.mu_0},\n'
                f'sigma={self.sigma})')


class NaiveBayesClassifier:
    def __init__(self) -> None:
        # prior and conditional probabilities
        self.pi = None  # pi_k
        self.p = None  # p_jkl

    def train(self, X: Array, y: Array) -> None:
        # X: feature matrix
        # y: labels
        # fit NBC by calculating MLE of prior and conditional probabilities
        # calculate prior probabilities
        self.pi = np.bincount(y) / len(y)
        # create tensor of one-hot encoded features
        X_ijl = torch.nn.functional.one_hot(torch.tensor(X).long())
        # create matrix of one-hot encoded labels
        y_ik = torch.nn.functional.one_hot(torch.tensor(y).long())
        # calculate conditional probabilities
        p_lkj = torch.einsum('ijl,ik->lkj', X_ijl, y_ik)
        # normalize conditional probabilities and permute dimensions
        p_jkl = (p_lkj / p_lkj.sum(dim=0)).permute(2, 1, 0).numpy()
        self.p = p_jkl

    def predict(self, X: Array) -> Array:
        # X: feature matrix
        # return: predicted labels
        # create tensor of one-hot encoded features
        X_ijl = torch.nn.functional.one_hot(torch.tensor(X).long()).numpy()
        # calculate posterior probabilities
        log_pi_ik = np.einsum('jkl,ijl->ik', np.log(self.p), X_ijl) + np.log(self.pi)
        return np.argmax(log_pi_ik, axis=1)

    def accuracy(self, X: Array, y: Array) -> float:
        # X: feature matrix
        # y: labels
        # return: accuracy
        return np.mean(self.predict(X) == y).round(5)

    def __str__(self) -> str:
        return (f'NaiveBayesClassifier(\n'
                f'pi={self.pi},\n'
                f'p={self.p})')


class KernelLogisticRegression:
    def __init__(self) -> None:
        self.alpha = None
        self.X_repr = None
        self.lamb = None
        self.kernel_type = None

    @staticmethod
    def linear_kernel(X, Y):
        # compute the linear kernel matrix
        # inputs: X: n x p data matrix
        #         Y: m x p data matrix
        # returns: K: n x n kernel matrix
        return X @ Y.T

    @staticmethod
    def rbf_kernel(X, Y, gamma=None):
        # compute the RBF kernel matrix
        # inputs: X: n x p data matrix
        #         Y: m x p data matrix
        #         gamma: kernel width
        # returns: K: n x m kernel matrix
        if gamma is None:
            gamma = 1 / X.shape[1]
        n = X.shape[0]
        m = Y.shape[0]
        X_sq = np.sum(X ** 2, axis=1)
        Y_sq = np.sum(Y ** 2, axis=1)
        X_sq = np.tile(X_sq, (m, 1)).T
        Y_sq = np.tile(Y_sq, (n, 1))
        K = X_sq + Y_sq - 2 * X @ Y.T
        return np.exp(- gamma * K)

    @staticmethod
    def poly_kernel(X, Y, degree=2, gamma=None, c=1):
        # compute the polynomial kernel matrix
        # inputs: X: n x p data matrix
        #         Y: m x p data matrix
        #         degree: degree of the polynomial
        #         gamma: scaling factor
        #         c: constant term
        # returns: K: n x m kernel matrix
        if gamma is None:
            gamma = 1 / X.shape[1]
        K = (gamma * (X @ Y.T) + c) ** degree
        return K

    @staticmethod
    def log_likelihood(K: Tensor, y: Tensor, alpha: Tensor) -> Tensor:
        # K: kernel matrix
        # y: labels
        # alpha: parameters
        # return: log likelihood
        return torch.sum(y * (K @ alpha) - torch.log(1 + torch.exp(K @ alpha)))

    def loss(self, K: Tensor, y: Tensor, alpha: Tensor, lamb: float) -> Tensor:
        # X: feature matrix
        # y: labels
        # alpha: parameters
        # lamb: penalty parameter
        # return: loss
        return - self.log_likelihood(K, y, alpha) + lamb * torch.sum(alpha ** 2)

    def train(self, X: Array, y: Array, kernel: str = 'linear', lamb: float = 0.01) -> None:
        # X: feature matrix
        # y: labels
        # kernel: kernel function
        # lamb: penalty parameter
        # fit KLR model
        # calculate kernel matrix
        if kernel == 'linear':
            K = self.linear_kernel(X, X)
            self.kernel_type = 'linear'
        elif kernel == 'rbf':
            K = self.rbf_kernel(X, X)
            self.kernel_type = 'rbf'
        elif kernel == 'poly':
            K = self.poly_kernel(X, X)
            self.kernel_type = 'poly'
        else:
            raise ValueError('Invalid kernel function')
        # save X for prediction
        self.X_repr = X
        # convert data to tensors
        K = torch.tensor(K).float()
        y = torch.tensor(y).float()
        # initialize parameters
        alpha = torch.zeros(K.shape[0], requires_grad=True)
        # train model
        optimizer = torch.optim.Adam([alpha], lr=0.005)
        for _ in range(1000):
            optimizer.zero_grad()
            loss = self.loss(K, y, alpha, lamb)
            loss.backward()
            optimizer.step()
        # save parameters
        self.alpha = alpha.detach().numpy()
        self.lamb = lamb

    def predict(self, X: Array) -> Array:
        # X: feature matrix
        # return: predicted labels
        if self.kernel_type == 'linear':
            K = self.linear_kernel(X, self.X_repr)
        elif self.kernel_type == 'rbf':
            K = self.rbf_kernel(X, self.X_repr)
        elif self.kernel_type == 'poly':
            K = self.poly_kernel(X, self.X_repr)
        else:
            raise ValueError('Invalid kernel function')
        return (1 + np.exp(-K @ self.alpha) < 2).astype(int)

    def accuracy(self, X: Array, y: Array) -> float:
        # X: feature matrix
        # y: labels
        # return: accuracy
        return np.mean(self.predict(X) == y).round(5)

    def cv_train(self, X: Array, y: Array, lambdas: [float], k: int, kernel: str = 'linear') -> None:
        # X: feature matrix
        # y: labels
        # lamb: penalty parameter
        # k: number of folds
        # kernel: kernel function
        # fit KLR model using cross-validation
        CV_accuracies = []
        for lamb in lambdas:
            test_accuracy = 0
            # split data into k folds
            n = len(y)
            fold_size = n // k
            for i in range(k):
                # split data into train and test sets
                test_indices = list(range(i * fold_size, (i + 1) * fold_size))
                train_indices = list(set(range(n)) - set(test_indices))
                X_tr, y_tr = X[train_indices], y[train_indices]
                X_te, y_te = X[test_indices], y[test_indices]
                # train model
                self.train(X_tr, y_tr, kernel, lamb)
                # calculate accuracy
                test_accuracy += self.accuracy(X_te, y_te)
            # calculate average accuracy
            CV_accuracies.append(test_accuracy / k)
        # select lambda with highest accuracy
        self.lamb = lambdas[np.argmax(CV_accuracies)]
        # train model using selected lambda
        self.train(X, y, kernel, self.lamb)

    def __str__(self) -> str:
        return (f'KernelLogisticRegression(\n'
                f'kernel_type={self.kernel_type})')


if __name__ == '__main__':
    # import train data from spam-data
    spam_data_train = pd.read_csv('./data/spam-data/spam-train.txt')
    spam_data_test = pd.read_csv('./data/spam-data/spam-test.txt')
    # split data into X and y
    X_train = spam_data_train.iloc[:, :-1].to_numpy()
    y_train = spam_data_train.iloc[:, -1].to_numpy()
    X_test = spam_data_test.iloc[:, :-1].to_numpy()
    y_test = spam_data_test.iloc[:, -1].to_numpy()
    # centering
    centered_X_train, centered_X_test = center_columns(X_train), center_columns(X_test)
    # log transformation
    log_X_train, log_X_test = log_transform(X_train), log_transform(X_test)
    # discrete transformation
    discrete_X_train, discrete_X_test = discrete_transform(X_train), discrete_transform(X_test)

    # part a
    # train penalized logistic regression
    # print(f"""{"="*70}\n"""
    #       f"""Penalized Logistic Regression\n"""
    #       f"""{"-"*70}""")
    # plr = PenalizedLogisticReg()
    # plr.cv_train(X_train, y_train, [0.0000001, 0.000001, 0.00001], 10)
    # print(f"Train accuracy on data: {round(plr.accuracy(X_train, y_train) * 100, 3)}%\n"
    #       f"Test accuracy on data: {round(plr.accuracy(X_test, y_test) * 100, 3)}%")
    # plr.cv_train(log_X_train, y_train, [0.1, 1, 10], 10)
    # print(f"Train accuracy on log transformed data: {round(plr.accuracy(log_X_train, y_train) * 100, 3)}%\n"
    #       f"Test accuracy on log transformed data: {round(plr.accuracy(log_X_test, y_test) * 100, 3)}%")

    # part b
    # train LDA classifier
    print(f"""{"="*70}\n"""
          f"""Linear Discriminant Analysis Classifier\n""" 
          f"""{"-"*70}""")
    lda = LDAClassifier()
    lda.train(centered_X_train, y_train)
    print(f"Train accuracy on centered data: {round(lda.accuracy(centered_X_train, y_train) * 100, 3)}%\n"
          f"Test accuracy on centered data: {round(lda.accuracy(centered_X_test, y_test) * 100, 3)}%")
    lda.train(log_X_train, y_train)
    print(f"Train accuracy on log transformed data: {round(lda.accuracy(log_X_train, y_train) * 100, 3)}%\n"
          f"Test accuracy on log transformed data: {round(lda.accuracy(log_X_test, y_test) * 100, 3)}%")

    # part c issue some features show only one level
    print(f"""{"="*70}\n"""
          f"""Naive Bayes Classifier\n""" 
          f"""{"-"*70}""")
    nbc = NaiveBayesClassifier()
    nbc.train(discrete_X_train, y_train)
    print(f"Train accuracy on discrete transformed data: {round(nbc.accuracy(discrete_X_train, y_train) * 100, 3)}%\n"
          f"Test accuracy on discrete transformed data: {round(nbc.accuracy(discrete_X_test, y_test) * 100, 3)}%")

    # part d
    print(f"""{"="*70}\n"""
          f"""Kernel Logistic Regression\n""" 
          f"""{"-"*70}""")
    klr = KernelLogisticRegression()
    klr.cv_train(centered_X_train, y_train, [0.00001, 0.0001, 0.001], 10, 'rbf')
    print(klr.lamb)
    print(f"Train accuracy on centered data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(centered_X_train, y_train) * 100, 3)}%\n"
          f"Test accuracy on centered data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(centered_X_test, y_test) * 100, 3)}%")
    klr.cv_train(log_X_train, y_train, [0.00001, 0.0001, 0.001], 10, 'rbf')
    print(klr.lamb)
    print(f"Train accuracy on log transformed data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(log_X_train, y_train) * 100, 3)}%\n"
          f"Test accuracy on log transformed data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(log_X_test, y_test) * 100, 3)}%")
    klr.cv_train(centered_X_train, y_train, [0.00001, 0.0001, 0.001], 10, 'poly')
    print(klr.lamb)
    print(f"Train accuracy on centered data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(centered_X_train, y_train) * 100, 3)}%\n"
          f"Test accuracy on centered data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(centered_X_test, y_test) * 100, 3)}%")
    klr.cv_train(log_X_train, y_train, [0.00001, 0.0001, 0.001], 10, 'poly')
    print(klr.lamb)
    print(f"Train accuracy on log transformed data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(log_X_train, y_train) * 100, 3)}%\n"
          f"Test accuracy on log transformed data with {klr.kernel_type} kernel: "
          f"{round(klr.accuracy(log_X_test, y_test) * 100, 3)}%")
