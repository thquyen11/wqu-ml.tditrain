import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_california_housing


data = fetch_california_housing()
X=data['data']
y=data['target']


# Simple Ridge Regression
ridge=Ridge(alpha=0.1, fit_intercept=True, normalize=True)
ridge.fit(X,y)
y_pred=ridge.predict(X)
R_squared=ridge.score(X, y)


# Pipeine Linear Regression
stdScaler=StandardScaler()
poly_features=PolynomialFeatures(degree=2)
lin_reg=LinearRegression()

pipe=Pipeline([
    ('scaler', stdScaler),
    ('poly', poly_features),
    ('regressor', lin_reg)
])
y_pred = pipe.predict(X)
R_squared=pipe.score(X, y)


# Pipeline Linear Regaression with Feature Union
stdScaler = StandardScaler()
pca=PCA(n_components=4)
selector=SelectKBest(f_regression, k=2)

pca_pipe=Pipeline([
    ('scaler', stdScaler),
    ('dim_red', pca)
])
union=FeatureUnion([
    ('pca_pipe', pca_pipe),
    ('selector', selector)
])
pipe=Pipeline([
    ('union', union),
    ('regressor', lin_reg)
])
pipe.fit(X,y)
R_squared=pipe.score(X, y)
print(R_squared)

