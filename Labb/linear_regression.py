import numpy as np

class LinearRegression:
    def __init__(self):
        self.b = None
        self.d = None
        self.n = None

    def fit(self, X, Y):
        self.n = X.shape[0]
        self.d = X.shape[1] - 1
        XT = X.T
        self.b = np.linalg.inv(XT @ X) @ XT @ Y

    def predict(self, X):
        return X @ self.b
    
    def sse(self, X, Y):
        residuals = Y - self.predict(X)
        return np.sum(residuals**2)
    
    def variance(self, X, Y):
        return self.sse(X, Y) / (self.n - self.d - 1)
    
    def std(self, X, Y):
        return np.sqrt(self.variance(X, Y))
    
    def rmse(self, X, Y):
        return np.sqrt(self.sse(X, Y) / self.n)
