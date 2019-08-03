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
