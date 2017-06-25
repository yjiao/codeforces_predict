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
from keras.layers.core import Masking
from keras.layers import Merge
from keras.regularizers import l2

from os.path import exists

# feature scaling has been removed from this version--we can't
# scale each sample by itself, every sample must be done together
java = [ 'Java 6', 'Java 8', 'Java 8 ZIP', 'Java 7' ]
python = [ 'Python 3', 'Python 2', 'PyPy 2', 'PyPy 3']
lowlevel = [ 'GNU C++14', 'GNU C', 'MS C++', 'GNU C++', 'GNU C++0x', 'MS C#', 'GNU C11', 'GNU C++11 ZIP', 'GNU C++11', 'Mono C#', 'Delphi', ]
otherlang = [ 'Haskell', 'Factor', 'Picat', 'Secret_171', 'PHP', 'Tcl', 'Scala', 'Io', 'FPC', 'J', 'Rust', 'JavaScript', 'Ada', 'Go', 'Cobol', 'Befunge', 'Roco', 'Ruby', 'Kotlin', 'F#', 'Perl', 'Pike', 'D', 'Ocaml' ]
errors = [ "COMPILATION_ERROR", "RUNTIME_ERROR", "CRASHED", "REJECTED", "IDLENESS_LIMIT_EXCEEDED"]
wrong = [ "TIME_LIMIT_EXCEEDED", "WRONG_ANSWER", "CHALLENGED", "MEMORY_LIMIT_EXCEEDED" ]
drop = [ 'FALSE', 'SKIPPED', 'TESTING', 'PARTIAL', 'REJECTED', 'PRESENTATION_ERROR', 'FAILED', 'rank' ]
practice = [ 'GYM', 'OUT_OF_COMPETITION', 'VIRTUAL' ]

def get_categorical_variables( colnames ):
    con = psycopg2.connect(database='codeforces', user='Joy')
    cur = con.cursor()
    catvars = []
    for c in colnames:
        cur.execute("select * from %s" % c, con)
        catvars.extend([c[1] for c in cur.fetchall()])

    return catvars

def create_model(n_neurons, batch_input_shape):
    model = Sequential()

    model.add(Masking(mask_value=0, batch_input_shape = batch_input_shape))

    model.add(GRU(
        n_neurons[0],
        return_sequences=True,
        stateful=True,
        batch_input_shape=batch_input_shape,
        dropout=0.5,
        use_bias=False
    ))

    model.add(GRU(
        n_neurons[1],
        return_sequences=False,
        stateful=True,
        batch_input_shape=batch_input_shape,
        dropout=0.5,
        use_bias=False
    ))

    model.add(Dense(
        n_neurons[2],
        activation='tanh',
        use_bias=False
    ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear', use_bias=False))
    return model

def compress_columns(cols, newname, data):
    data[newname] = np.sum(data[cols], axis=1)

def get_user_data(user, binvars, month, maxtimepts, columns, maxtime):
    y_column = 'delta_smoothed_%dmonths' % month

    # -----------------------------
    # Load data
    data = pd.read_csv('rnn_train/%s.csv'%user)

    # ABORT if file is empty
    if data.shape[0] == 0:
        return None, None, None, None

    # reorder columns: flat files columns are mixed bc they were made from python dict
    data = data[columns]

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
    data.loc[idx1, y_column] = data.loc[idx2, y_column]

    # -----------------------------
    # binarize some variables
    data[binvars] = data[binvars].fillna(value=0)
    for b in binvars:
        data.loc[ data[b] > 0, b] = 1


    # compress languages
    compress_columns(java, 'java', data)
    compress_columns(lowlevel, 'lowlevel', data)
    compress_columns(python, 'python', data)
    compress_columns(otherlang, 'otherlang', data)
    compress_columns(errors, 'errors', data)
    compress_columns(wrong, 'wrong', data)
    compress_columns(practice, 'practice', data)
    data.drop(binvars + errors + wrong + drop, axis=1, inplace=True)


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

    # -----------------------------
    # drop columns we don't need
    df_train.drop("ratingupdatetimeseconds", axis=1, inplace=True)
    df_train.drop("solvetimeseconds", axis=1, inplace=True)
    df_train.drop("starttimeseconds", axis=1, inplace=True)
    df_train.drop("stoptimeseconds",  axis=1, inplace=True)
    df_train.drop("newrating",  axis=1, inplace=True)
    df_train.drop("problemid",  axis=1, inplace=True)
    #df_train.drop("oldrating",  axis=1, inplace=True)

    # -----------------------------
    # create list of inputs for training
    trainlist = []
    ylist = []
    colnames = [d for d in df_train.columns.values if d != 'contestid' and d != y_column]

    # -----------------------------
    # Clip time columns, some people did problems arbitrarily long in the past
    hourcols = ['hours_solve_to_contest', 'hours_submit_to_contest', 'hours_submit_to_solve']
    df_train[hourcols] = np.clip(df_train[hourcols], 0, maxtime)
    
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
    n_features = trainlist[0].shape[1]
    maxtimepts_actual = max(map(len, trainlist))

    size = trainlist[0].shape[1]
    for i in range(len(trainlist)):
        gap = maxtimepts - len(trainlist[i])
        if (gap < 0):
            trainlist[i] = trainlist[i][0:maxtimepts,:]
        else:
            zeros = np.zeros((gap, size))
            trainlist[i] = np.concatenate([trainlist[i], zeros], axis=0)

    arx = np.concatenate(trainlist, axis=0)
    arx = np.reshape(arx, (len(trainlist), maxtimepts, n_features))
    
    return arx, ary, maxtimepts_actual, colnames 

def get_train_data(handles, binvars, correct_cols, maxtimepts, path, maxtime):
    X = []
    Y = []
    maxt = 0
    lens = [0] * len(handles)
    runsum = 0

    for i in range(len(handles)):
        filename = path + handles[i] + ".csv"
        print filename

        if not exists(filename):
            continue

        x, y, t, colnames = get_user_data(
            handles[i],
            binvars=binvars,
            month=3,
            maxtimepts=maxtimepts,
            columns=correct_cols,
            maxtime=maxtime)

        if x is None:
            continue

        maxt = max(t, maxt)
        X.append(x)
        Y.append(y)
        runsum += x.shape[0]
        lens[i] = runsum

    return X, Y, lens, maxt, colnames

def wide_and_deep(n_neurons, wideshape, deepshape, verbose=False):
    # ---------------------------------
    # Wide network for binary variables
    # ---------------------------------
    wide = Sequential()
    wide.add(Conv1D(
        filters=n_neurons[2],
        kernel_size=wideshape[1],
        batch_input_shape=wideshape
    ))
    wide.add(GlobalAveragePooling1D())

    # ---------------------------------
    # Deep network for dense variables
    # ---------------------------------
    deep = Sequential()
    deep.add(Masking(mask_value=0, batch_input_shape = deepshape))

    deep.add(GRU(
        n_neurons[0],
        return_sequences=True,
        stateful=False,
        batch_input_shape=deepshape,
        dropout=0.5,
        use_bias=False
    ))

    deep.add(GRU(
        n_neurons[1],
        return_sequences=False,
        stateful=True,
        batch_input_shape=deepshape,
        dropout=0.5,
        use_bias=False
    ))

    deep.add(Dense(n_neurons[2], activation='tanh', use_bias=False))
    deep.add(Dropout(0.5))

    # ---------------------------------
    # Combine Wide and Deep
    # ---------------------------------
    model = Sequential()
    model.add(Merge([wide, deep], mode='concat', concat_axis=1))
    model.add(Dense(1, activation='tanh', use_bias=False))

    if verbose:
        print wide.summary()
        print deep.summary()
        print model.summary()

    return model
