#Generate features for OLS

import psycopg2
import pandas as pd
import numpy as np
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
practice = [ 'GYM', 'OUT_OF_COMPETITION', 'VIRTUAL', 'PRACTICE' ]

def get_categorical_variables( colnames ):
    con = psycopg2.connect(database='codeforces', user='Joy')
    cur = con.cursor()
    catvars = []
    for c in colnames:
        cur.execute("select * from %s" % c, con)
        catvars.extend([c[1] for c in cur.fetchall()])

    return catvars

def compress_columns(cols, newname, data):
    data[newname] = np.sum(data[cols], axis=1)

def binarize_variables(data, binvars):
    for b in binvars:
        data.loc[ data[b] > 0, b] = 1

    compress_columns(java, 'java', data)
    compress_columns(lowlevel, 'lowlevel', data)
    compress_columns(python, 'python', data)
    compress_columns(otherlang, 'otherlang', data)
    compress_columns(errors, 'errors', data)
    compress_columns(wrong, 'wrong', data)
    compress_columns(practice, 'practice', data)
    data.drop(binvars + errors + wrong + drop + practice, axis=1, inplace=True)

def remove_nonrelevant_months(data, month=3):
    # remove information for other months
    for m in range(1,6):
        if m == month:
            continue
        name1 = "delta_smoothed_%dmonths" % m
        name2 = "smoothed_%dmonths" % m
        
        data.drop([name1, name2], axis=1, inplace=True)

def get_user_data(data, binvars, month, columns):
    y_column = 'delta_smoothed_%dmonths' % month

    # -----------------------------
    # binarize variables
    data.fillna(value=0, inplace=True)
    binarize_variables(data, binvars)

    # -----------------------------
    # drop the first contest--we don't have a "real" change from a null expectation here
    # assign all activity to the second contest
    data.sort_values('ratingupdatetimeseconds', inplace=True)
    cids = list(set(data.contestid.values))
    
    # ABORT if only one contest
    if len(cids) == 1:
        return None

    cids.sort()
    cid1 = cids[0]
    cid2 = cids[1]

    idx1 = data.index[data.contestid == cid1]
    idx2 = data.index[data.contestid == cid2][0]

    data.loc[idx1, 'ratingupdatetimeseconds'] = data.loc[idx2, 'ratingupdatetimeseconds']
    data.loc[idx1, 'contestid'] = data.loc[idx2, 'contestid']
    data.loc[idx1, y_column] = data.loc[idx2, y_column]


    data.drop(['index', 'newrating', 'CONTESTANT'], axis=1, inplace=True)

    # -----------------------------
    # remove unncessary months
    remove_nonrelevant_months(data, month)

    return data

def agg_by_window():
    pass

def get_features(handle, binvars, correct_cols, path='rnn_train/', month=3):
    # -----------------------------
    # Load data
    filename = path + handle + ".csv"
    if not exists(filename):
        return None

    data = pd.read_csv(filename)
    
    # ABORT if file is empty
    if data.shape[0] == 0:
        return None

    # reorder columns: flat files columns are mixed bc they were made from python dict
    data = data[correct_cols]

    x = get_user_data(
        data,
        binvars,
        month=month,
        columns=correct_cols)

    return x

