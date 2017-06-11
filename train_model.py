import pandas as pd
import numpy as np
from random import randint

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import GRU 

from sklearn.metrics import r2_score
from keras_functions import *

from os.path import exists
binvars = ['PRACTICE', 'OUT_OF_COMPETITION', 'GYM', 'VIRTUAL', 'CONTESTANT', 'Java 6', 'Mysterious Language', 'GNU C++14', 'FALSE', 'Haskell', 'Delphi', 'GNU C', 'Python 3', 'Factor', 'Picat', 'MS C++', 'Secret_171', 'PHP', 'Tcl', 'Java 8', 'Scala', 'Io', 'Python 2', 'GNU C++', 'FPC', 'J', 'Rust', 'PyPy 2', 'JavaScript', 'GNU C++0x', 'MS C#', 'Ada', 'Go', 'GNU C11', 'Cobol', 'Befunge', 'Roco', 'Ruby', 'Kotlin', 'GNU C++11 ZIP', 'F#', 'GNU C++11', 'Perl', 'Pike', 'Java 8 ZIP', 'D', 'Ocaml', 'PyPy 3', 'Mono C#', 'Java 7']

def make_model(batchsize, weights=None, load=None):
    # ------------------------------------------
    # Define model
    # ------------------------------------------
    gru1 = 6
    gru2 = 3
    dense = 50
    model = create_model([gru1, gru2, dense], batchsize)
    if load is not None:
        model.load_weights(load)
    if weights is not None:
        model.set_weights(weights)

    # ------------------------------------------
    # Compile
    # ------------------------------------------
    optimizer = keras.optimizers.RMSprop(lr=0.01, decay=0.01)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

def train_sample(handle, correct_cols, xmax, ymax, maxtimepts, maxtime, weights=None, load=None, epochs=50):
    print "========================================\ntraining %s, %d epochs" % (handle, epochs)
    # ------------------------------------------
    # Get data
    # ------------------------------------------
    X, Y, _, _, colnames = get_train_data([handle], binvars, correct_cols, maxtimepts, path, maxtime)

    batchsize = X[0].shape
    model = make_model(batchsize, weights=weights, load=load)

    # ------------------------------------------
    # Train
    # ------------------------------------------
    history = []
    xin = X[0]/xmax
    yin = Y[0]/ymax

    for i in range(epochs):
        h = model.fit(xin, yin, epochs=1, shuffle=False, batch_size=batchsize[0], verbose=0)
        model.reset_states()
        history.append(h)

    write_to_file(history)
    model.save_weights('model_weights.h5')
    return history, model.get_weights()

def write_to_file(history):
    loss = np.concatenate([h.history['loss'] for h in history])
    mae = np.concatenate([h.history['mean_absolute_error'] for h in history])

    print "   final loss: %f, mae: %f\n" % (loss[-1], mae[-1])
    with open("training_loss.txt", 'a') as f:
	for e in zip(loss, mae):
	    f.write(",".join(map(str, e)) + "\n")

if __name__ == '__main__':
    # ------------------------------------------
    # parameters
    # ------------------------------------------
    month = 3
    tags = set(['implementation', 'two pointers', 'data structures', '*special', 'probabilities', 'divide and conquer', 'shortest paths', 'meet-in-the-middle', 'trees', 'matrices', 'graph matchings', 'expression parsing', 'graphs', 'ternary search', 'dfs and similar', 'combinatorics', 'string suffix structures', 'games', 'binary search', '2-sat', 'brute force', 'hashing', 'dsu', 'chinese remainder theorem', 'flows', 'sortings', 'number theory', 'fft', 'greedy', 'schedules', 'math', 'strings', 'bitmasks', 'geometry', 'dp', 'constructive algorithms'])
    ymax = 500.0
    maxrating = 5000.0
    maxpoints = 3000.0
    maxtimepts = 100
    maxtime = 24 * 30 * month * 2
    path = 'rnn_train/'

    # ------------------------------------------
    # Get list of handles in train/ test/ val
    # ------------------------------------------
    with open('set_train.txt') as f:
        train_handles = [h.strip() for h in f.readlines()]
    with open('set_test.txt') as f:
        test_handles = [h.strip() for h in f.readlines()]
    with open('set_val.txt') as f:
        val_handles = [h.strip() for h in f.readlines()]

    # user columns can be out of order, use tourist's as correct order
    with open(path + 'tourist.csv') as f:
        correct_cols = [t.strip() for t in f.readline().split(',')]
        correct_cols = [c for c in correct_cols if c not in tags]

#    # ----------------------------------------------------------
#    # Modify flatfiles into padded numpy ndarrays for training
#    # ----------------------------------------------------------
#    # randomly sample 10 people to get the max
#    idx = [randint(0, len(train_handles)) for i in range(10)]
#    X, Y, _, maxt, colnames = get_train_data(np.array(train_handles)[idx], binvars, correct_cols, maxtimepts, path)
#
#    # ------------------------------------------
#    # Scaling
#    # ------------------------------------------
#    # note here we are simply concattentating all of X and setting a max on that
#    Xflat = np.concatenate(X)
#    Xflat = np.reshape(Xflat, [Xflat.shape[0]*Xflat.shape[1], Xflat.shape[2]])
#
#    xmax = np.max(Xflat, axis=0)
#    xmax[xmax == 0] = 1
#    # normalization vector
#    # set rating columns on the same scale
#    colnames_rate = ['smoothed_%dmonths'%month, 'oldrating', 'problem_rating']
#    maxrating = 5000.0
#    idx_rate = [colnames.index(c) for c in colnames_rate]
#    xmax[idx_rate] = maxrating
    xmax = np.array([
            maxtimepts,
            maxtime,
            maxtime,
            maxtime,
            maxrating,
            maxpoints,
            maxrating,
            maxrating,
            maxtimepts,
            maxtimepts,
            maxtimepts,
            maxtimepts,
            maxtimepts,
            maxtimepts,
            maxtimepts
            ])

    print list(xmax)

    # ------------------------------------------
    # Iterate through training set
    # ------------------------------------------
    #histories = []
    if exists('model_weights.h5'):
        _, weights = train_sample(train_handles[0], correct_cols, xmax, ymax, maxtimepts, maxtime, load='model_weights.h5')
    else:
        _, weights = train_sample(train_handles[0], correct_cols, xmax, ymax, maxtimepts, maxtime)

    for i, handle in enumerate(train_handles[1:]):
        print i, handle
        if not exists(path + handle + ".csv"):
            continue
        _, weights = train_sample(handle, correct_cols, xmax, ymax, maxtimepts, maxtime, weights=weights)
