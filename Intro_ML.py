import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


if __name__=='__main__':
    X = np.linspace(0,1,100)
    exp=np.random.choice([2,3])
    y=X**exp+np.random.randn(X.shape[0])/10
    plt.plot(X,y,'.')
    