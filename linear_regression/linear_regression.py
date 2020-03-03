import numpy as np
import matplotlib.pyplot as plt

class linear_regression(object):
    def __init__(
            self,
            predictors,
            X,
            Y,
            X_test,
            Y_test,
            standardize=True,
            intercept=True,
            weighted=False):
        '''
            Initalize linear regression object
            with the dataset
            X = (P+1/P, N) dim matrix
            Y = N dim vector
            input: Standardize [bool](whiten data)
            intercept [bool](Include intercept in model)
        '''
        if standardize and intercept:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
            predictors.append('intercept')
        elif standardize and not intercept:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
            Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
        elif not standardize and intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            predictors.append('intercept')
        # initalized
        self.predictors = predictors
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test


    def solve(self, solver='Simple', weighted=False, verbose = False):
        '''
            Solve regression directly or 
            with Successive Gram-Schmidt Orthonormalization 
            solver:  "Simple", 'gs' [str] (solution method)
            weighted [bool](Weighted OLS)
            returns a dataframe with 
        '''
        if solver == 'Simple':
            ret = self.solve_simple()
        elif solver == 'gs':
            ret = gs_solver()
        RSS = self._RSS()
        rsq = self.r_squared()
        self.z_score = self.zscore()
        if verbose:
            print(f"***** {solver} Least-Squares Estimate")
            print(
                '{:<10.10s}{:<10.8s}{:<10.8s}{:<10.8s}'.format(
                    "Predictor",
                    "Coef.",
                    "Std. err.",
                    "Z-score"))

            dash = '-' * 40
            print(dash)
            for i in range(len(self.predictors)):
                print(
                    '{:<10.8s}{:>10.3f}{:>10.3f}{:>10.3f}'.format(
                        self.predictors[i],
                        self.beta[i][0],
                        self.beta[i][0],
                        self.z_score[i]))
            print(f"***** R^2: {rsq}")

    def solve_simple(self, weighted=False):
        '''
            Direct least-squares solution.
            b = (XtX)^-1XTY
            yhat = xb
        '''
        self.invxtx = np.linalg.inv(self.X.T @ self.X) 
        beta = self.invxtx @ self.X.T @ self.Y
        y_hat = self.X @ beta
        self.beta = beta
        self.y_hat = y_hat
        return beta, y_hat

    def gs_solver(self, weighted=False):
        '''
            Gram-Schmidt Orthogonalization:
            using QR decomposition
            Note: QR decomp required Np^2 operations 
            R - Upper traingular matrix
            Q - Note: np.linalg.qr is doing the heavy lifting
        '''
        Q, R = np.linalg.qr(self.X)
        self.beta = np.linalg.inv(R) @ Q.T @ self.Y
        self.y_hat = Q.T @ Q @ self.U
        return self.beta, self.y_hat

    def zscore(self, weighted=False):
        '''
            Z-score
            For the jth predictor
            z_j = beta_hat_j / (sqrt(var * v_j))
            where v = (X.T*X)_jj 
        '''
        v = np.diag(self.invxtx)
        var = self._var()
        return np.ravel(self.beta)/(np.sqrt(var)*np.sqrt(v)) 

    def _RSS(self, weighted=False):
        '''
            Multivariate RSS Calculation
        '''
        self.rss = None
        err = self.Y - self.y_hat
        if weighted:
            self.rss = np.trace(err.T @ np.cov(err) @ err) 
        else:
            self.rss = np.trace(err.T @ err)
        return self.rss 

    def r_squared(self):
        '''
            Multivariate RSS Calculation
        '''
        if self.rss:
            tss = np.sum((self.Y - np.mean(self.Y))**2)
            return 1 - self.rss/tss
        else:
            return None

    def _var(self, weighted=False):
        '''
            Returns an unbiased estimate of the sample variance sigma^2
            sigma^1 = 1/(N-p-1) * MSE
        '''
        N, p = self.X.shape
        return 1 / (N - p - 1) * np.sum((self.Y - self.y_hat)**2)


    def pred_error(self, beta):
        y_pred = self.X_test @ beta 
        return np.sum((y_pred - self.Y_test)**2)


    def backwards_stepwise_selection(self):
        '''
            returns a list of variables dropped during each iterations
        '''
        import copy
        #regress on the full model, then drop the predictor with the smallest
        #z-score 
        x_prev, y_prev = copy.deepcopy(self.X), copy.deepcopy(self.Y)
        x_test_prev  = copy.deepcopy(self.X_test)
        pred_prev = copy.deepcopy(self.predictors)
        rssarr, p_dropped = [], []
        prederr  = []
        for i in range(len(self.predictors)-1):
            self.solve()
            min_idx = np.argmin(np.abs(self.z_score))
            p_dropped.append(self.predictors[min_idx])
            rssarr.append(self.rss)
            prederr.append(self.pred_error(self.beta))
            #delete column
            self.X = np.delete(self.X,min_idx,axis=1)
            self.X_test = np.delete(self.X_test,min_idx,axis=1)
            self.predictors = np.delete(self.predictors,min_idx,axis=0)

        self.X = x_prev
        self.Y = y_prev
        self.X_test = x_test_prev
        self.predictors = pred_prev

        return p_dropped, prederr


