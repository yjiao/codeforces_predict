import psycopg2
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU

def get_categorical_variables( colnames ):
    con = psycopg2.connect(database='codeforces', user='Joy')
    cur = con.cursor()
    catvars = []
    for c in colnames:
        cur.execute("select * from %s" % c, con)
        catvars.extend([c[1] for c in cur.fetchall()])

    return catvars

def get_user_data(user, binvars, month, maxtimepts):
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
    data[binvars] = data[binvars].fillna(value=0)

    for b in binvars:
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
    y_column = 'delta_smoothed_%dmonths' % month
    colnames = list(df_train.columns.values)

    cids = df_train.contestid

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train_scaled = scaler.fit_transform(df_train)
    df_train_scaled = pd.DataFrame(df_train_scaled)
    df_train_scaled.columns = colnames

    # add back in colnames that should not be scaled
    df_train_scaled['contestid'] = cids
    df_train_scaled[y_column] = df_train[y_column]
    
    # scale the y column wrt to max change possible
    df_train_scaled[y_column] /= max_change

    # -----------------------------
    # Group by contest
    groups = df_train_scaled.groupby('contestid')

    # -----------------------------
    # create list of inputs for training
    trainlist = []
    ylist = []

    for k, v in groups:
        v.is_copy = False
        
        v.drop('contestid', axis=1, inplace=True)
        y = v.loc[:, y_column].values[0]
        v.drop(y_column, inplace=True, axis=1)
        
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
