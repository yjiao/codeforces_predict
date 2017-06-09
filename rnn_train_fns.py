# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

# ---------------- Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM, GRU

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import psycopg2
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
np.random.seed(7)

# create sql connections
con = psycopg2.connect(database='codeforces', user='Joy')
cur = con.cursor()

## -----------------------------
## HYPERPARAMETERS
## -----------------------------
#month = 3
#bins = range(-200, 200, 20)
#maxtimepts = 50
#n_neurons1 = 10

def get_user_data(user):
    # -----------------------------
    # Load data
    data = pd.read_csv('rnn_train/%s.csv'%user)

    # drop the first contest--too much variance
    data.sort_values('ratingupdatetimeseconds', inplace=True)
    firstcid = data.contestid.values[0]
    data.drop(data.index[data.contestid == firstcid], axis=0, inplace=True)

    # -----------------------------
    # binarize some variables

    cur.execute("select * from all_participanttypes", con)
    all_part = [c[1] for c in cur.fetchall()]

    cur.execute("select * from all_tags", con)
    all_tags = [c[1] for c in cur.fetchall()]

    cur.execute("select * from all_language", con)
    all_lang = [c[1] for c in cur.fetchall()]

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
    # Feature scaling and grouping by contest
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


    yvecs = [ [0] * (len(bins) + 1) for i in range(len(ylist))]
    idx1 = np.digitize(ylist, bins=bins)
    for i, j in enumerate(idx1):
        yvecs[i][j] = 1
    ary = np.array(yvecs)


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


## -----------------------------
## Set up keras model
## -----------------------------
## keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
#arx, ary = get_user_data('chenmark')
#arx_test, ary_test = get_user_data('tourist')
#
##maxtimepts = arx.shape[1]
#n_features = arx.shape[2]
#batchsize = arx.shape[0]
#
#model = Sequential()
##model.add(Embedding(max_features, output_dim=256))
#batch_input_shape = (batchsize, maxtimepts, n_features)
#model.add(LSTM(n_neurons1, return_sequences=False, stateful=True, batch_input_shape=batch_input_shape))
#model.add(Dropout(0.5))
##model.add(LSTM(32, return_sequences=False, stateful=True))
##model.add(Dropout(0.5))
#model.add(Dense(len(bins) + 1, activation='softmax'))
#
#print "batch input shape ", batch_input_shape
#print "model output shape", model.output_shape
#
## For a multi-class classification problem
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#
#
#print "input x shape", arx.shape
#print "input y shape", ary.shape
#
## fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
#model.fit(arx, ary, epochs=50, batch_size=batchsize)
#score = model.evaluate(arx_test, ary_test, batch_size=arx_test.shape[0])
#
#print score
