import numpy as np
import matplotlib.pyplot as plt


class linear_regression(object):
    def __init__(
            self,
            predictors,
            X,
            Y,
            standardize=True,
            intercept=True,
            weighted=False):
        '''
            Initalize linear regression object
            with the dataset
            X = (P+1, N) dim matrix
            Y = N vector
        '''
        if standardize and intercept:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            #Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        elif standardize and not intercept:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
        elif not standardize and intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        # initalized
        self.predictors = predictors
        self.X = X
        self.Y = Y

    def solve(self, solver='Simple', weighted=False):
        if solver == 'Simple':
            ret = self.solve_simple()
            RSS = self._RSS()
        elif solver == 'gs':
            ret = gs_solver()
            RSS = self._RSS()

        print(f"***** {solver} Least-Squares Estimate")
        print(
            '{:<10.8s}{:<10.8s}{:<10.8s}{:<10.8s}'.format(
                "Predictor",
                "Coef.",
                "Std. err.",
                "Z-score"))

        dash = '-' * 40
        print(dash)
        for i in range(len(self.predictors)):
            print(
                '{:<10.8s}{:>10.3f}{:>10.3f}{:>10.3f}'.format(
                    self.predictors[i][0],
                    self.beta[i][0],
                    self.beta[i][0],
                    self.beta[i][0]))
        print(f"***** RSS")
        print(RSS)

    def solve_simple(self, weighted=False):
        '''
        Direct least-squares solution
        '''
        beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        y_hat = self.X @ beta
        self.beta = beta
        self.y_hat = y_hat
        return beta, y_hat

    def gs_solver(self, weighted=False):
        '''
        Gram-Schmidt Orthogonalization:
        using QR decomposition
        R - Upper traingular matrix
        Q - Note: np.linalg.qr is doing the heavy lifting '''
        Q, R = np.linalg.qr(self.X)
        self.beta = np.linalg.inv(R) @ Q.T @ self.Y
        self.y_hat = Q.T @ Q @ self.U
        return self.beta, self.y_hat

    def _RSS(self, weighted=False):
        '''
        Multivariate RSS Calculation
        '''
        err = self.Y - self.y_hat
        if weighted:
            return np.trace(err.T @ np.cov(err) @ err)
        else:
            return np.trace(err.T @ err)

    def _var(self, weighted=False):
        '''
        Returns an unbiased estimate of the sample variance sigma^2
        sigma^1 = 1/(N-p-1) * MSE
        '''
        N, p = X.shape
        return 1 / (N - p - 1) * np.sum((self.Y - self.y_hat)**2)
