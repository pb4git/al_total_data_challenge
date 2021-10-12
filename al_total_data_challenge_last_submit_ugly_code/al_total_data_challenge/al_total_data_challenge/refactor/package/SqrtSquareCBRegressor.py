from sklearn.base import BaseEstimator
import numpy as np

class SqrtSquareCBRegressor(BaseEstimator):
    def __init__(self, regr):
        self.regr = regr

    def fit(self, X, y):
        self.regr.fit(X, np.sqrt(y), cat_features=list(X.select_dtypes(include='category').columns))
        return self

    def predict(self, X):
        return np.clip(self.regr.predict(X), 0, 1000)**2
   
from sklearn.preprocessing import QuantileTransformer
class QuantileCBRegressor(BaseEstimator):
    def __init__(self, regr):
        self.regr = regr
        self.qt = QuantileTransformer(subsample=1e6)

    def fit(self, X, y):
        self.qt.fit(y.values.reshape(-1, 1))
        self.regr.fit(X, self.qt.transform(y.values.reshape(-1, 1)), cat_features=list(X.select_dtypes(include='category').columns))
        return self

    def predict(self, X):
        return self.qt.inverse_transform(np.clip(self.regr.predict(X), 0, 1).reshape(-1, 1))
    
    
from sklearn.preprocessing import PowerTransformer
class PowerCBRegressor(BaseEstimator):
    def __init__(self, regr):
        self.regr = regr
        self.qt = PowerTransformer()

    def fit(self, X, y):
        self.qt.fit(y.values.reshape(-1, 1))
        self.regr.fit(X, self.qt.transform(y.values.reshape(-1, 1)), cat_features=list(X.select_dtypes(include='category').columns))
        return self

    def predict(self, X):
        return np.clip(self.qt.inverse_transform(self.regr.predict(X).reshape(-1, 1)), 0, 0.95)
    
    
class SqrtSquareLGBMRegressor(BaseEstimator):
    def __init__(self, regr):
        self.regr = regr

    def fit(self, X, y):
        self.regr.fit(X, np.sqrt(y), categorical_feature=list(X.select_dtypes(include='category').columns))
        return self

    def predict(self, X):
        return np.clip(self.regr.predict(X), 0, 1000)**2