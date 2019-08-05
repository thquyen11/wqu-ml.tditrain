import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.datasets import fetch_california_housing


# Simple Linear Regression
data = fetch_california_housing()
X=data['data']
y=data['target']
ridge=Ridge(alpha=0.1)
ridge.fit(X, y)


# Pipeline without Union Feature
stdScaler=StandardScaler()
poly=PolynomialFeatures(degree=2)
pipe=Pipeline([
    ('scaler', scaler),
    ('poly', poly),
    ('regressor', ridge)
])
pipe.fit(X,y)
y_pred=pipe.predict(X)
R_squared=pipe.score(X, y)


# Pipeline with Union Feature
pca = PCA(n_components=4)
selector = SelectKBest(f_regression, k=2)
pca_pipe = Pipeline([('scaler', scaler), ('dim_red', pca)])
union = FeatureUnion([('pca_pipe', pca_pipe), ('selector', selector)])
pipe = Pipeline([('union', union), ('regressor', lin_reg)])
pipe.fit(X, y)
y_pred=pipe.predict(X)
R_squared=pipe.score(X, y)


# Custom estimator
class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, q_lower, q_upper):
        self.q_lower=q_lower
        self.q_upper=q_upper

    def fit(self, X, y=None):
        self.upper=np.percentile(X, self.q_upper, axis=0)
        self.lower=np.percentile(X, self.q_lower, axis=0)
        return self

    def transform(self, X):
        Xt=X.copy()
        idx_lower=X < self.lower
        idx_upper=X>self.upper
        for i in range(X.shape[-1]):
            Xt[idx_lower[:, i], i] = self.lower[i]
            Xt[idx_upper[:, i], i] = self.upper[i]
        
        return Xt

replacer = OutlierReplacer(5, 95)
replacer.fit(X)
Xt=replacer.transform(X)


# Custom Regressor
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.y_mean=np.mean(y)
        return self

    def predict(self, X):
        return self.y_mean*np.ones(X.shape[0])

mean_regressor=MeanRegressor()
mean_regressor.fit(X, y)
y_pred=mean_regressor.predict(X)
    
