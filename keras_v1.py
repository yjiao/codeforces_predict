# coding: utf-8

# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import psycopg2
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from keras_functions import *

# -----------------------------------------
# HYPER PARAMETERS
# -----------------------------------------

month = 3

# fix random seed for reproducibility
np.random.seed(7)

con = psycopg2.connect(database='codeforces', user='Joy')
cur = con.cursor()

# -----------------------------------------
# LOAD SQL QUERIES
# -----------------------------------------
# binarize some variables
all_part, all_tags, all_lang = get_categorical_variables([
    'all_participanttypes',
    'all_tags',
    'all_language'
])

# -----------------------------------------
# Set up keras model
# -----------------------------------------

def get_user_data(user):
    # set some parameters for normalization
    max_change = 200.0
    
    # -----------------------------
    # Load data
    data = pd.read_csv('rnn_train/%s.csv'%user)

    # drop the first contest--we don't have a "real" change from a null expectation here
    data.sort_values('ratingupdatetimeseconds', inplace=True)
    firstcid = data.contestid.values[0]
    data.drop(data.index[data.contestid == firstcid], axis=0, inplace=True)

    # -----------------------------
    # binarize some variables
    # set binary columns to binary, some of them were counts by mistake
    bin_vars = all_part + all_tags + all_lang
    data[bin_vars] = data[bin_vars].fillna(value=0)

    for b in bin_vars:
        data.loc[ data[b] > 0, b] = 1

    # -----------------------------
    # remove information for other months
    df_data = data
    for m in range(1,6):
        if m == month:
            continue
        name1 = "delta_smoothed_%dmonths" % m
        name2 = "smoothed_%dmonths" % m
        
        df_data.drop([name1, name2], axis=1, inplace=True)

    df_train = df_data.drop(['handle', 'index'], axis=1)
    df_train.fillna(value=0, inplace=True)


    # -----------------------------
    # Feature scaling
    colname = 'delta_smoothed_%dmonths' % month
    cols = list(df_train.columns.values)
    colidx = cols.index(colname)

    cids = df_train.contestid

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train_scaled = scaler.fit_transform(df_train)
    df_train_scaled = pd.DataFrame(df_train_scaled)
    df_train_scaled.columns = cols

    # add back in cols that should not be scaled
    df_train_scaled['contestid'] = cids
    df_train_scaled[colname] = df_train[colname]
    
    # scale the y column wrt to max change possible
    df_train_scaled[colname] /= max_change

    # -----------------------------
    # Group by contest
    groups = df_train_scaled.groupby('contestid')

    # -----------------------------
    # create list of inputs for training
    trainlist = []
    ylist = []

    for k, v in groups:
        base = [0] * (len(bins) + 1)
        v.is_copy = False
        
        v.drop('contestid', axis=1, inplace=True)
        y = v.loc[:, colname].values[0]
        v.drop(colname, inplace=True, axis=1)
        
        trainlist.append(v)
        ylist.append(y)

    ary = np.array(ylist)

    # -----------------------------
    # Pad X values
    # TODO: need to make this "universal" across all users
    #maxtimepts = max([len(t) for t in trainlist])
    size = trainlist[0].shape[1]

    for i in range(len(trainlist)):
        gap = maxtimepts - len(trainlist[i])
        for j in range(gap):
            nullrow = [0] * size
            trainlist[i].loc[-j-1] = nullrow
        trainlist[i].sort_index(inplace = True)


    dfx = pd.concat(trainlist)
    dfx.reset_index(inplace=True, drop=True)

    arx = np.array(dfx)
    arx = np.reshape(arx, (len(trainlist), maxtimepts, 111))
    return arx, ary

xx_train, yy_train = get_user_data("lewin")
xx_test, yy_test = get_user_data("chenmark")

# -----------------------------------------
# keras parameters
maxtimepts = 110 # number of time points before a contest to sample
size = 111 # number of parameters

neurons1 = 10
neurons2 = 4

def create_model(layer1, layer2, batch_input_shape, maxtimepts):
    model = Sequential()
    batch_input_shape = (batch_input_shape[0], maxtimepts, size)
    model.add(LSTM(layer1, return_sequences=True, stateful=True, batch_input_shape=batch_input_shape))
    model.add(Dropout(0.5))
    model.add(LSTM(layer2, return_sequences=False, stateful=True, batch_input_shape=batch_input_shape))
    model.add(Dropout(0.5))
    # we use a linear activation for the output
    model.add(Dense(1, activation='linear'))
    return model

model = create_model(neurons1, neurons2, xx_train.shape)

print model.summary()

optimizer = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['mae'])

history = model.fit(xx_train, yy_train, epochs=50, batch_size=xx_train.shape[0], shuffle=False)

#plt.rcParams['figure.figsize'] = (10.0, 10.0)
#plt.plot(history.history['mean_absolute_error'])

newmodel = create_model(neurons1, neurons2, xx_train.shape)
old_weights = model.get_weights()
newmodel.set_weights(old_weights)
newmodel.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mae'])

score = newmodel.evaluate(xx_test, yy_test, batch_size=xx_test.shape[0])
score

y_pred_train = model.predict(xx_train, batch_size=xx_train.shape[0])
y_pred_test = newmodel.predict(xx_test, batch_size=xx_test.shape[0])

from sklearn.metrics import r2_score
print r2_score(yy_train, y_pred_train)
print r2_score(yy_test, y_pred_test)
