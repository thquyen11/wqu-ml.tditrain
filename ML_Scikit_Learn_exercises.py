#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.datasets import fetch_california_housing

#%%
data=fetch_california_housing()
df = pd.DataFrame(data['data'], columns=data['feature_names'])


#%%
# Linear Regression without customized transformer
X=df
y=data['target']

lr=LinearRegression(fit_intercept=True, normalize=True)
lr.fit(X, y)
R_squared=lr.score(X, y)
print(R_squared)


#%%
# Linear Regression with customized transformer that calculate the distance away from Lost Angeles and San Francissco
class Distance(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, X1_lat, X1_long, X2_lat, X2_long):
        return ((X2_lat-X1_lat)**2 + (X2_long-X1_long)**2)**(1/2)

LosAngeles_coord=[34.052235, -118.243683]
SanFrancisco_coord=[37.733795, -122.446747]
distance=Distance()

dist = np.array([])
for i in range(len(df.index.values)):
    dist = np.append(dist, distance.fit_transform(LosAngeles_coord[0], LosAngeles_coord[1], df['Latitude'].iloc[i], df['Longitude'].iloc[i]))
df['distance_Los'] = dist

dist = np.array([])
for i in range(len(df.index.values)):
    dist = np.append(dist, distance.fit_transform(SanFrancisco_coord[0], SanFrancisco_coord[1], df['Latitude'].iloc[i], df['Longitude'].iloc[i]))
df['distance_San'] = dist

stdScaler=StandardScaler()
poly=PolynomialFeatures(degree=2)
pipe=Pipeline([
    ('scaler', stdScaler),
    ('poly', poly),
    ('regressor', lr)
])

X = df
y = data['target']
pipe.fit(X,y)
y_pred=pipe.predict(X)
R_squared=pipe.score(X, y)


#%%
# Serialize the model
import dill
with open('my_model.dill', 'wb') as f:
    dill.dump(pipe, f)
    


#%%
