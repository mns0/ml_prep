import numpy as np
import matplotlib.pyplot as plt


class linear_regression(object):
    def __init__(self, X, Y, standardize=False, intercept=True):
        '''
            Initalize linear regression object
            with the dataset
            X = (P+1, N) dim matrix
            Y = N vector
        '''
        if standardize:
            X = (X - np.mean(X, axis=1)) / np.std(X, axis=1)
            Y = (Y - np.mean(Y, axis=1)) / np.std(Y, axis=1)
        if intercept:
            X = np.hstack((X,np.ones((X.shape[0], 1))))
        # initalized
        self.X = X
        self.Y = Y

    def solve(self,solver='simple'):
        if solver == 'simple':
            ret = self.solve_simple()
            RSS = self._RSS()
        else:
            return None
        return ret

    def solve_simple(self):
        beta =  np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y
        y_hat = self.X @ beta
        self.beta = beta
        self.y_hat = y_hat
        return beta, y_hat


    def gs_solver(self):
        pass


    def _RSS(self): if self.beta and self.y_hat:
            y_bar = np.mean(self.Y)
            return 1- np.sum((self.y_hat - y_bar)**2) / np.sum((self.Y - y_bar)**2) 


    def _RSS(self): if self.beta and self.y_hat:
            y_bar = np.mean(self.Y)
            return 1- np.sum((self.y_hat - y_bar)**2) / np.sum((self.Y - y_bar)**2)  
