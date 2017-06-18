# coding: utf-8
# Gridsearch for RF regressor

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import copy

# read train, validation, and test data from previously pickled file
with open('ensemble_data.pickle') as f:
    data = pickle.load(f)

df_train = data['xtrain']
df_val = data['xval']
#df_test = data['xtest']

y_train = data['ytrain']
y_val = data['yval']
#y_test = data['ytest']

# since the grid search takes care of cv for us, merge the train and validations sets
df_train = pd.concat([df_train, df_val])
y_train = pd.concat([y_train, y_val])

x = np.array(df_train)
y = np.array(y_train)

# define rf regressor ---------------------------------------
rf = RandomForestRegressor(random_state=12358, n_jobs=-1)
param_grid = dict(
        n_estimators = [10, 20, 50, 100, 500],
        min_samples_split = [2, 5, 10, 100],
        max_depth = [None, 10, 100, 1000]
        )

# define grid search --------------------------------------
#GridSearchCV(estimator,
#    param_grid,
#    scoring=None,
#    fit_params=None,
#    n_jobs=1,
#    iid=True,
#    refit=True,
#    cv=None,
#    verbose=0,
#    pre_dispatch='2*n_jobs',
#    error_score='raise',
#    return_train_score=True)
gsearch = GridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=2,
        return_train_score=True
        )

gsearch.fit(x, y)

with open('rfregressor_gridsearch_fit.pickle', 'w') as f:
    pickle.dump(gsearch, f)
