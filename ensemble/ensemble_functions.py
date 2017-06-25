from glob import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_train_val_test():
    files = glob('ols_train/*.csv')
    alldata = []
    for h in files:
        try:
            t = pd.read_csv(h, engine='c')
            alldata.append(t)
        except:
            pass
    alldata = pd.concat(alldata)

    # --------------------------------------
    # get previous features
    prev_train = pd.read_csv('training_linear_regression_smooth.csv', engine='c')
    df_smooth = pd.read_csv('user_ratings_smoothed_atleast5contests.csv', engine='c')

    mergedata = alldata
    mergedata.set_index(['handle', 'contestid'], inplace=True)
    prev_train.set_index(['handle', 'contestid'], inplace=True)
    df_smooth.set_index(['handle', 'contestid'], inplace=True)
    mergedata = mergedata.join(prev_train)

    # --------------------------------------
    # remove first contests for all users
    # there is no delta for here
    g = mergedata.groupby('handle')
    glist = []
    for k, v in g:
        glist.append(v.drop(v.index[v.starttimeseconds == min(v.starttimeseconds)], axis=0))
    mergedata = pd.concat(glist)

    X = mergedata
    Y = X['delta_smoothed_3months']
    dropcols = [
                'delta_smoothed_3months',
                'delta_smoothed_1months',
                'delta_smoothed_2months',
                'delta_smoothed_4months',
                'delta_smoothed_5months',
                'smoothed_1months',
                'rank',
                'smoothed_2months',
                'smoothed_4months',
                'smoothed_5months',
                'contestid',
                'type',
                'ratingupdatetimeseconds',
                'starttimeseconds',
                'newrating',
                "level_0",
                'index',
                'rating_change_smooth',
                'rating_change',
                'user_rating_smooth',
                'contesttime',
                'min_probability_practice',
                'min_probability_contest'
              ]
    for d in dropcols:
        if d in X.columns:
            X.drop(d, inplace=True, axis=1)

    return X, Y

# note that this version is for ensemble learning, and we are not using the smoothed version of delta
def get_data_for_ensemble():
    files = glob('ols_train/*.csv')
    alldata = []
    for h in files:
        try:
            t = pd.read_csv(h, engine='c')
            alldata.append(t)
        except:
            pass
    alldata = pd.concat(alldata)

    # --------------------------------------
    # get previous features
    prev_train = pd.read_csv('training_linear_regression_smooth.csv', engine='c')
    df_smooth = pd.read_csv('user_ratings_smoothed_atleast5contests.csv', engine='c')

    mergedata = alldata
    mergedata.set_index(['handle', 'contestid'], inplace=True)
    prev_train.set_index(['handle', 'contestid'], inplace=True)
    df_smooth.set_index(['handle', 'contestid'], inplace=True)
    mergedata = mergedata.join(prev_train)

    # --------------------------------------
    # remove first contests for all users
    # there is no delta for here
    g = mergedata.groupby('handle')
    glist = []
    for k, v in g:
        glist.append(v.drop(v.index[v.starttimeseconds == min(v.starttimeseconds)], axis=0))
    mergedata = pd.concat(glist)

    X = mergedata
    Y = X['newrating'] - X['oldrating']
    dropcols = [
                'delta_smoothed_1months',
                'delta_smoothed_2months',
                'delta_smoothed_3months',
                'delta_smoothed_4months',
                'delta_smoothed_5months',
                'contestid',
                'type',
                'ratingupdatetimeseconds',
                'starttimeseconds',
                'newrating',
                'user_rating',
                "level_0",
                'index',
                'rating_change_smooth',
                'rating_change',
                'user_rating_smooth',
                'contesttime',
                'min_probability_practice',
                'min_probability_contest'
              ]
    for d in dropcols:
        if d in X.columns:
            X.drop(d, inplace=True, axis=1)

    return X, Y