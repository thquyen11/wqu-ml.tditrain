import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


if __name__=='__main__':
    X = np.linspace(0,1,100)
    exp=np.random.choice([2,3])
    y=X**exp+np.random.randn(X.shape[0])/10
    plt.plot(X,y,'.')

    # Linear Regression
    lr=LinearRegression(fit_intercept=True, normalize=False)
    model_fit=lr.fit(X.reshape(-1,1), y)
    print(model_fit.coef_, model_fit.intercept_)
    y_pred=model_fit.predict(X.reshape(-1,1))
    plt.plot(X.reshape(-1,1), y, '.', label='data')
    plt.plot(X.reshape(-1,1), y_pred, label='model')
    plt.show()
