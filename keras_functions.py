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

# feature scaling has been removed from this version--we can't
# scale each sample by itself, every sample must be done together

def get_categorical_variables( colnames ):
    con = psycopg2.connect(database='codeforces', user='Joy')
    cur = con.cursor()
    catvars = []
    for c in colnames:
        cur.execute("select * from %s" % c, con)
        catvars.extend([c[1] for c in cur.fetchall()])

    return catvars

def get_user_data(user, binvars, month, maxtimepts):
    # -----------------------------
    # Load data
    data = pd.read_csv('rnn_train/%s.csv'%user)

    # ABORT if file is empty
    if data.shape[0] == 0:
        return None, None, None, None

    # drop the first contest--we don't have a "real" change from a null expectation here
    # assign all activity to the second contest
    data.sort_values('ratingupdatetimeseconds', inplace=True)
    cids = list(set(data.contestid.values))
    
    # ABORT if only one contest
    if len(cids) == 1:
        return None, None, None, None

    cids.sort()
    cid1 = cids[0]
    cid2 = cids[1]

    idx1 = data.index[data.contestid == cid1]
    idx2 = data.index[data.contestid == cid2][0]

    data.loc[idx1, 'ratingupdatetimeseconds'] = data.loc[idx2, 'ratingupdatetimeseconds']
    data.loc[idx1, 'contestid'] = data.loc[idx2, 'contestid']

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
    # Group by contest
    groups = df_train.groupby('contestid')
    y_column = 'delta_smoothed_%dmonths' % month

    # -----------------------------
    # Get the differences in times of interest
    # Otherwise normalization would make them too small to be meaningful

    # current quantities
    # ratingupdatetimeseconds    time at end of a contest
    # solvetimeseconds           time at first solve
    # starttimeseconds           time of first submit
    # stoptimeseconds            time of last submit

    # quantities of interest
    # hours_solve_to_contest      hours between solve and end of contest
    # hours_submit_to_contest     hours between first submit and end of contest
    # hours_submit_to_solve       hours between first submit and solve
    df_train['hours_solve_to_contest'] = (df_train['ratingupdatetimeseconds'] - df_train['solvetimeseconds']) / 3600.0
    df_train['hours_submit_to_contest'] = (df_train['ratingupdatetimeseconds'] - df_train['starttimeseconds']) / 3600.0
    df_train['hours_submit_to_solve'] = (df_train['solvetimeseconds'] - df_train['starttimeseconds']) / 3600.0

    # some problems were never solved. In this case hours_submit_to_solve is set to -1
    idx_neversolved = df_train['solvetimeseconds'] < 0
    df_train.loc[idx_neversolved, 'hours_solve_to_contest'] = -1
    df_train.loc[idx_neversolved, 'hours_submit_to_solve'] = -1

    df_train.drop("ratingupdatetimeseconds", axis=1, inplace=True)
    df_train.drop("solvetimeseconds", axis=1, inplace=True)
    df_train.drop("starttimeseconds", axis=1, inplace=True)
    df_train.drop("stoptimeseconds",  axis=1, inplace=True)

    # -----------------------------
    # create list of inputs for training
    trainlist = []
    ylist = []
    colnames = df_train.columns.values
    
    for k, v in groups:
        v.is_copy = False
        
        v.drop('contestid', axis=1, inplace=True)
        y = v.loc[:, y_column].values[0]
        v.drop(y_column, inplace=True, axis=1)
        
        trainlist.append(np.array(v))
        #trainlist.append(v)
        ylist.append(y)

    ary = np.array(ylist)

    # -----------------------------
    # Pad X values
    # TODO: need to make this "universal" across all users

    n_features = trainlist[0].shape[1]
    maxtimepts_actual = max(map(len, trainlist))

    size = trainlist[0].shape[1]
    for i in range(len(trainlist)):
        gap = maxtimepts - len(trainlist[i])
        zeros = np.zeros((gap, size))
        trainlist[i] = np.concatenate([trainlist[i], zeros], axis=0)

    arx = np.concatenate(trainlist, axis=0)
    arx = np.reshape(arx, (len(trainlist), maxtimepts, n_features))
    
    return arx, ary, maxtimepts_actual, colnames 

def create_model(n_neurons batch_input_shape):
    model = Sequential()

    model.add(Masking(mask_value=0, batch_input_shape = batch_input_shape))

    model.add(GRU(n_neurons[0], return_sequences=True, stateful=True, batch_input_shape=batch_input_shape))
    model.add(Dropout(0.5))

    model.add(GRU(n_neurons[1], return_sequences=False, stateful=True, batch_input_shape=batch_input_shape))
    model.add(Dropout(0.5))

    model.add(Dense(n_neurons[2], activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='tanh'))
    return model
