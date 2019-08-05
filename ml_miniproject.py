# %%
# import h2o4gpu as sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression, f_classif
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split


# %%
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')
fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')


############################################################################
# EXCERCISE 1
# %%
class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, grouper):
        self.grouper = grouper
        self.group_averages = {}

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        df = pd.DataFrame(X[self.grouper], columns=[self.grouper])
        df['y'] = y
        df = df.groupby([self.grouper]).mean()
        for index, row in df.iterrows():
            self.group_averages[index] = float(row.values)
        return self

    def predict(self, X):
        # Return a list of predicted penalties based on group of samples in X
        result = []
        if isinstance(X, list):
            X = pd.DataFrame(X)
        for i in X[self.grouper]:
            if i in self.group_averages:
                result.append(self.group_averages[i])
            else:
                result.append(np.mean(list(self.group_averages.values())))
        return result


# %%
state_model = Pipeline([
    ('sme', GroupMeanEstimator(grouper='STATE'))
])

state_model.fit(data, fine_totals)

# Predict a un-existing state
y_pred = state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))


################################################################################
# EXERCISE 2
# %%
simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF',
               'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']


class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]


# %%
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
])


# %%
class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].median()
                               if (X[c].dtype == 'int64' or X[c].dtype == 'float64') else X[c].mode().values[0] for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# %%
X_train, X_test, y_train, y_test = train_test_split(
    data, fine_counts > 0, test_size=0.3, random_state=42)

imputer = DataFrameImputer()
preprocess = Pipeline([('simple', simple_features),
                       ('imp', imputer),
                       ])
simple_features_model = Pipeline([
    ('preprocess', preprocess),
    ('classifier', LogisticRegression()),
])

gs = GridSearchCV(simple_features_model,
                  param_grid=[{'classifier': [LogisticRegression()],
                               'classifier__penalty': ['l1', 'l2'],
                               'classifier__C': np.logspace(-4, 4, 20),
                               'classifier__solver': ['liblinear']},
                              {'classifier': [RandomForestClassifier()],
                               'classifier__n_estimators': list(range(10, 100, 10)),
                               'classifier__max_features': list(range(1, 5, 2))}
                              ],
                  cv=5,
                  n_jobs=-1,
                  verbose=True)

gs.fit(X_train, y_train)
y_pred = gs.predict(X_test)
print('simple_features_model scores')
print('accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('roc_auc: ', metrics.roc_auc_score(y_test, y_pred))
print('estimator: ', gs.estimator)


# %%
best_model = [0, 0]
for i in range(5):
    rs = RandomizedSearchCV(simple_features_model,
                            {'classifier': [LogisticRegression()],
                             'classifier__penalty': ['l1', 'l2'],
                             'classifier__C': np.logspace(-4, 4, 20),
                             'classifier__solver': ['liblinear']},
                            cv=5,
                            n_jobs=-1,
                            scoring='neg_log_loss')
    rs.fit(X_train, y_train)
    y_pred = rs.predict(X_train)
    precision_score = metrics.precision_score(y_train, y_pred)
    if precision_score > best_model[0]:
        best_model[0] = precision_score
        best_model[1] = rs

rs = best_model[1]
y_pred = rs.predict(X_test)
print('simple_features_model scores')
print('accuracy: ', metrics.accuracy_score(y_test, y_pred))
print('roc_auc: ', metrics.roc_auc_score(y_test, y_pred))
print('estimator: ', gs.estimator)


###############################################################################
# EXERCISE 3
# %%
class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        integer_encoded = label_encoder.fit_transform(X[self.column])
        self.X_encoded = onehot_encoder.fit_transform(
            integer_encoded.reshape(-1, 1))
        return self

    def transform(self, X):
        return self.X_encoded


# %%
data_ex3 = data.drop(['EXP_TOTAL', 'ADJ_TOTAL'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    data_ex3, fine_counts > 0, test_size=0.3, random_state=42)


# %%
pca = PCA(n_components=4)
ctree = tree.DecisionTreeClassifier(criterion='entropy',
                                    max_depth=10, random_state=1)
owner_hot = Pipeline([('cst', ColumnSelectTransformer('OWNERSHIP'))])
cert_hot = Pipeline([('cst', ColumnSelectTransformer('CERTIFICATION'))])
categorical_features = Pipeline([('union', FeatureUnion([('owner_hot', owner_hot),
                                                         ('cert_hot', cert_hot)])),
                                 ('pca', pca)])

categorical_features_model = Pipeline([('categorical', categorical_features),
                                       #    ('clf', knn),
                                       ('clf', ctree),
                                       ])

categorical_features_model.fit(X_train, y_train)
print(metrics.classification_report(
    y_train, categorical_features_model.predict(X_train)))


# %%
rs = RandomizedSearchCV(categorical_features_model,
                        {'clf__max_depth': range(1, 30)},
                        # {'categorical__pca__n_components': range(1, 10),
                        #  'clf__max_depth': range(1, 30)},
                        cv=5,
                        n_jobs=7,
                        scoring='neg_log_loss')

rs.fit(data_ex3, fine_counts > 0)


# %%
