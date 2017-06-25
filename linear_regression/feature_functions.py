#Generate features for OLS

import psycopg2
import pandas as pd
import numpy as np
from os.path import exists

# ---------------------------------------------
# some hard-coded groupings of sparse features
# languages
java = [ 'Java 6', 'Java 8', 'Java 8 ZIP', 'Java 7' ]
python = [ 'Python 3', 'Python 2', 'PyPy 2', 'PyPy 3']
lowlevel = [ 'GNU C++14', 'GNU C', 'MS C++', 'GNU C++', 'GNU C++0x', 'MS C#', 'GNU C11', 'GNU C++11 ZIP', 'GNU C++11', 'Mono C#', 'Delphi', ]
otherlang = [ 'Haskell', 'Factor', 'Picat', 'Secret_171', 'PHP', 'Tcl', 'Scala', 'Io', 'FPC', 'J', 'Rust', 'JavaScript', 'Ada', 'Go', 'Cobol', 'Befunge', 'Roco', 'Ruby', 'Kotlin', 'F#', 'Perl', 'Pike', 'D', 'Ocaml' ]
# verdicts
errors = [ "COMPILATION_ERROR", "RUNTIME_ERROR", "CRASHED", "REJECTED", "IDLENESS_LIMIT_EXCEEDED"]
wrong = [ "TIME_LIMIT_EXCEEDED", "WRONG_ANSWER", "CHALLENGED", "MEMORY_LIMIT_EXCEEDED" ]
# practice mdoes
practice = [ 'GYM', 'OUT_OF_COMPETITION', 'VIRTUAL', 'PRACTICE' ]
# columns that are too sparse or reduandant to be informative
drop = [ 'FALSE', 'SKIPPED', 'TESTING', 'PARTIAL', 'REJECTED', 'PRESENTATION_ERROR', 'FAILED', 'rank' ]

# --------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------
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

def get_unique_contests(data):
    contest_idx = data.CONTESTANT == 1
    cids = set(data.contestid[contest_idx])
    
    indices = []
    for c in cids:
        idx = min(data.loc[np.logical_and(contest_idx, data.contestid==c)].index.values)
        indices.append(idx)
    indices.sort()
    return indices

def get_stats_df(df, column, filter_):
    df = df[column][filter_]
    mean = np.mean(df)
    max_ = np.max(df)
    tot  = np.sum(df)
    std = np.std(df)
    return (mean, max_, tot, std)

def get_stats_col(col):
    mean = np.mean(col)
    max_ = np.max(col)
    min_ = np.min(col)
    tot  = np.sum(col)
    std = np.std(col)
    return (mean, max_, min_, tot, std)

def remove_nonrelevant_months(data, month=3):
    # remove information for other months
    for m in range(1,6):
        if m == month:
            continue
        name1 = "delta_smoothed_%dmonths" % m
        name2 = "smoothed_%dmonths" % m
        
        data.drop([name1, name2], axis=1, inplace=True)

# --------------------------------------------------------
# MAIN FUNCTIONS
# --------------------------------------------------------
def parse_data(data, binvars, month, columns):
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


    data.drop(['index', 'newrating'], axis=1, inplace=True)

    # -----------------------------
    # remove unncessary months
    remove_nonrelevant_months(data, month)

    return data

def agg_by_window(data, position, window=3, month=3): #window is in months
    # tags??
    # hacks?
#    - ave, mean, max, min amount of time spent on problem
#    - bool for whether problem was solved
#    - prob of solving this problem given rating
#    - number of unlikely solves
#    - rate of change over time
#    - number of tags not previously solved by user (vs popularity of tags in contest)


    # columns in df:
    #['CONTESTANT', # 'contestid', # 'delta_smoothed_3months', # 'handle', # 'hours_solve_to_contest',
    # 'hours_submit_to_contest', # 'hours_submit_to_solve', # 'oldrating', # 'points', # 'problem_rating',
    # 'problemid', # 'ratingupdatetimeseconds', # 'smoothed_3months', # 'solvetimeseconds', # 'starttimeseconds', # 'stoptimeseconds',
    # 'java', # 'lowlevel', # 'python', # 'otherlang', # 'errors', # 'wrong', # 'practice']

    # apply a sliding window approach to the data
    # instead of looking at problems done in the last n months, we look at the last n problems practiced
    # we can modify things based on the time variable involved
    
    # note that this df is sorted by time
    end = data.loc[position,'starttimeseconds']
    start = end - (window * 30 * 24 * 3600)
    #print "%d: (%d, %d)"%(position, start, end)

    idxfilter = np.logical_and(data.starttimeseconds >= start, data.starttimeseconds < end)
    dfw = data.loc[idxfilter]
    dfw.is_copy = False
    dfw.sort_values('starttimeseconds', inplace=True)

    n_problems_solved = dfw.shape[0]

    # --------------------------------------------------------------------
    # number of wrong problems
    # num problems wrong in practice
    mean_wrong_contest,\
        max_wrong_contest,\
        total_wrong_contest,\
        std_wrong_contest = get_stats_df(dfw, 'wrong', dfw.CONTESTANT > 0)
    # num problems wrong in contest
    mean_wrong_practice,\
        max_wrong_practice,\
        total_wrong_practice,\
        std_wrong_practice = get_stats_df(dfw, 'wrong', dfw.practice > 0)

    # --------------------------------------------------------------------
    # number of errors
    # num problems errors in practice
    mean_errors_contest,\
        max_errors_contest,\
        total_errors_contest,\
        std_errors_contest = get_stats_df(dfw, 'errors', dfw.CONTESTANT > 0)
    # num problems errors in contest
    mean_errors_practice,\
        max_errors_practice,\
        total_errors_practice,\
        std_errors_practice = get_stats_df(dfw, 'errors', dfw.practice > 0)

    # --------------------------------------------------------------------
    # difference between user rating and problem rating
    rating_column = 'smoothed_%dmonths' % month
    dfw['ratingdiff'] = dfw['problem_rating'] - dfw[rating_column]
    mean_ratingdiff_contest,\
            max_ratingdiff_contest, _,\
            std_ratingdiff_contest = get_stats_df(dfw, 'ratingdiff', dfw.CONTESTANT > 0)
    mean_ratingdiff_practice,\
            max_ratingdiff_practice, _,\
            std_ratingdiff_practice = get_stats_df(dfw, 'ratingdiff', dfw.practice > 0)

    # --------------------------------------------------------------------
    # Time between solves
    timediff = dfw.starttimeseconds - np.roll(dfw.starttimeseconds, 1)
    timediff = timediff[1:] / 3600.0 / 24.0
    # (mean, max_, min_, tot, std)
    mean_timediff, max_timediff, min_timediff, _, std_timediff = get_stats_col(timediff)

    # --------------------------------------------------------------------
    # Time between first submit and solves
    # (mean, max_, min_, tot, std)
    mean, max_, min_, tot, std = get_stats_col(dfw.hours_submit_to_solve)
    print max_
    print dfw.loc[dfw.hours_submit_to_solve == max_, ['contestid', 'problemid', 'starttimeseconds', 'stoptimeseconds']]

    # --------------------------------------------------------------------
    # num problems solved > threshold
    # (mean, max_, min_, tot, std)
    _, _, _, n100, _ = get_stats_col(dfw.ratingdiff >= 100)
    _, _, _, n200, _ = get_stats_col(dfw.ratingdiff >= 200)
    _, _, _, n300, _ = get_stats_col(dfw.ratingdiff >= 300)
    _, _, _, n400, _ = get_stats_col(dfw.ratingdiff >= 400)
    _, _, _, n500, _ = get_stats_col(dfw.ratingdiff >= 500)

def calc_Pcontest():
    # the likelihood of user's performance on a contest
    pass

def get_features(handle, binvars, correct_cols, path='rnn_train/', month=3):
    # -----------------------------
    # Load data
    filename = path + handle + ".csv"
    if not exists(filename):
        return None

    data = pd.read_csv(filename)
    # sort by time a problem was first attempted
    data.sort_values('starttimeseconds', inplace=True)
    data.reset_index(inplace=True)
    
    # ABORT if file is empty
    if data.shape[0] == 0:
        return None

    # reorder columns: flat files columns are mixed bc they were made from python dict
    data = data[correct_cols]

    x = parse_data(
        data,
        binvars,
        month=month,
        columns=correct_cols)

    idx = get_unique_contests(data)
    for i in idx:
        agg_by_window(x, i)

    return x

