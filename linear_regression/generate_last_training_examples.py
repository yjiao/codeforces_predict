# This script is made for demonstrating the UI + linear regression
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import psycopg2
from os.path import exists

db = 'codeforces'
usr = 'Joy'
con = psycopg2.connect(database = db, user = usr)
cur = con.cursor()

cur.execute("""SELECT handle FROM handles;""")
handles = [x[0] for x in cur.fetchall()]

q = """ SELECT * FROM problem_rating """
problem_ratings = pd.read_sql(q, con)
valid_contests = set(problem_ratings.contestid)

problem_ratings.set_index(['contestid', 'problemid'], inplace=True)

def get_problemstats(df, usr_rating):
    g = df.groupby(('contestid', 'problemid'))
    nsolved = 0
    wrong = []
    ratings = []
    solvetime = []
    n_prac = 0
    n_cont = 0
    
    # if we can't estimate ratings for a problem, for now just ignore them
    # (or--assume it's the same as the median of other problems that were done during this period??)
    for k, v in g:
        n_cont += 'CONTESTANT' in v.participanttype.values
        nsolved += 'OK' in v.verdict.values
        temp = list(v.verdict.values)
        wrong.append(len(v.verdict) - temp.count('OK'))
        
        try:
            solvetime.append(min(v.loc[v.verdict == 'OK', 'starttimeseconds']))
        except:
            pass
        
        try:
            ratings.append(problem_ratings.loc[k, 'problemrating'])
        except:
            pass
    solvetime.sort()
    timebetween = np.diff(solvetime)
    harder = sum(ratings > usr_rating)
    harder50 = sum(ratings > (usr_rating + 50))
    harder100 = sum(ratings > (usr_rating + 100))
    harder500 = sum(ratings > (usr_rating + 500))
    rating_diff = ratings - usr_rating
    
    return {'problems_solved': nsolved, # total number of problems solved
            'n_wrong_mean': np.mean(wrong),
            'n_wrong_std': np.std(wrong), # mean var of number of wrong tries on problem
            'rating_diff_mean': np.mean(rating_diff),
            'rating_diff_std': np.std(rating_diff), # mean, var of difference between user rating and problem rating
            'time_between_mean': np.mean(timebetween),
            'time_between_std': np.std(timebetween),
            'n_harder': harder, # number of harder problems than user rating
            'n_harder50': harder50,
            'n_harder100': harder100,
            'n_harder500': harder500,
            'n_contest': n_cont,
            'user_rating': usr_rating
           }

def get_training_data(submissions, user_rating, max_time_elapsed):
    data = []
    
    cid = user_rating.contestid
    ctime = user_rating.ratingupdatetimeseconds
    try:
	usr_rating_new = user_rating.loc[int(cid), 'newrating']
        df_train = submissions

	x = get_problemstats(df_train, usr_rating_new)

	data.append(x)
    except:
	pass

    return data

# params
filename = 'training_linear_regression_last.csv'
maxtime = 30 * 24 * 3600 # number of seconds in a month
lastidx = 0

print "Getting last contests for all users..."
q = """
SELECT *
FROM
    (
    SELECT *,
           max(ratingupdatetimeseconds) over (partition by handle) maxtime
    FROM user_rating
    )a
"""
last_contests = pd.read_sql(q, con)
print "done"

time_back = 3600*5 # 5 hrs

# get data AFTER the last contest
for k, series_usrrate in last_contests.iterrows():
    df_usrsub = pd.read_sql(""" SELECT * FROM submissions
        WHERE
            handle = '%s'
            AND
            starttimeseconds >= %d
        ; """ % (series_usrrate.handle, series_usrrate.ratingupdatetimeseconds - time_back),
        con)

    if series_usrrate.contestid not in valid_contests:
	print series_usrrate.contestid, valid_contests
	continue

    data = get_training_data(df_usrsub, series_usrrate, maxtime)
    data = pd.DataFrame.from_dict(data)
    data['handle'] = h

    if exists(filename):
        data.to_csv(filename, mode='a', header=False, index=False)
    else:
        data.to_csv(filename, mode='a', header=True, index=False)
    break