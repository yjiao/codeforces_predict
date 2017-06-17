import numpy as np
import pandas as pd
import psycopg2
from collections import defaultdict
from os.path import exists

# current to-do:
# keep those with only 1 contests?
# keep last trailing activities?

def get_categorical_variables( colnames, cur):
    catvars = []
    for c in colnames:
        cur.execute("select * from %s" % c, con)
        catvars.extend([c[1] for c in cur.fetchall()])

    return catvars

def getTraining(user, user_rating, problem_rating, hdl_cid_pid_list, binvars, con, df_tags):
    filename = 'rnn_train/%s.csv' % user
    print filename
    trainlist = []
    if len(user_rating.shape) == 1:
        print "     Not enough contests for user", user_rating.shape
        return
    user_rating.is_copy = False
    user_rating.reset_index(inplace=True)
    
    for hdl_cid_pid in hdl_cid_pid_list:
        q = """
        SELECT * FROM submissions
            WHERE
                handle = '%s'
                AND
                contestid = '%s'
                AND
                problemid = '%s'
        """ % (hdl_cid_pid[0], hdl_cid_pid[1], hdl_cid_pid[2])

        df_user_problem = pd.read_sql(q, con)
        df_user_problem.is_copy = False

        ex = dict()
        
        # generic problem info
        ex['points'] = df_user_problem.points[0]
        ex['problemid'] = df_user_problem.problemid.values[0]
        ex['contestid'] = df_user_problem.contestid.values[0]
        
        # ----------------------------------
        # user rating info
        # find closest PREVIOUS contest
        # if there is no next contest,then skip this entry
        ex['starttimeseconds'] = min(df_user_problem.starttimeseconds)
        ex['stoptimeseconds'] = max(df_user_problem.starttimeseconds)

        # all contests greater than submit time
        idx_prevcontest = user_rating.ratingupdatetimeseconds <= ex['starttimeseconds']
        idx_nextcontest = user_rating.ratingupdatetimeseconds >= ex['stoptimeseconds']
        
        # if this is the first few submissions before the user has competed,
        # then we set the "current rating" to the next contest rating instead
        if not np.any(idx_prevcontest):
            user_rating_contest = user_rating.loc[idx_nextcontest]
            # take min of all contests greater than submit time--this is the next contest
            idx_nextcontest = user_rating_contest.ratingupdatetimeseconds == min(user_rating_contest.ratingupdatetimeseconds)
            user_rating_contest = user_rating_contest.loc[idx_nextcontest].to_dict(orient='records')[0]
        else:
            user_rating_contest = user_rating.loc[idx_nextcontest]
            idx_prevcontest = user_rating_contest.ratingupdatetimeseconds == max(user_rating_contest.ratingupdatetimeseconds)
            user_rating_contest = user_rating_contest.loc[idx_prevcontest].to_dict(orient='records')[0]

        user_rating_contest['next_contestid'] = user_rating_contest['contestid']
        user_rating_contest.pop('contestid', None)
        user_rating_contest.pop('index', None)
        ex.update(user_rating_contest)

        # ----------------------------------
        # verdicts
        vcnt = df_user_problem.verdict.value_counts()
        vdict = vcnt.to_dict()
        ex.update(vdict)

        # participant type
        pcnt = df_user_problem.participanttype.value_counts()
        pdict = pcnt.to_dict()
        for t in pdict.iterkeys():
            ex[t] = 1

        # ----------------------------------
        # language
        lcnt = df_user_problem.language.value_counts()
        ldict = lcnt.to_dict()
        ex.update(ldict)

        # ----------------------------------
        # problem rating
        if (hdl_cid_pid[1], hdl_cid_pid[2]) in problem_rating.index:
            ex['problem_rating'] = problem_rating.loc[hdl_cid_pid[1], hdl_cid_pid[2]].values[0]
        else:
            ex['problem_rating'] = -1

        # ----------------------------------
        # time to solves
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
        solvetime = df_user_problem.loc[df_user_problem.verdict=='OK', 'starttimeseconds']

        ex['hours_submit_to_contest'] = (ex['ratingupdatetimeseconds'] - ex['starttimeseconds']) / 3600.0

        if len(solvetime) > 0:
            ex['solvetimeseconds'] = min(solvetime)
            ex['hours_solve_to_contest'] = (ex['ratingupdatetimeseconds'] - ex['solvetimeseconds']) / 3600.0
            ex['hours_submit_to_solve'] = (ex['solvetimeseconds'] - ex['starttimeseconds']) / 3600.0
        else:
            # some problems were never solved
            ex['solvetimeseconds'] = -1
            ex['hours_solve_to_contest'] = -1
            ex['hours_submit_to_solve'] = -1

        # -----------------------------------------------------------------------
        # tags
        idx_tag = (ex['contestid'], ex['problemid'])
        if (idx_tag) in df_tags:
            for t in df_tags.loc[idx_tag]['tag'].values:
                ex[t] = 1

        trainlist.append(ex)


    df_train = pd.DataFrame.from_dict(trainlist)
    for t in binvars:
        if t not in df_train.columns:
            df_train[t] = np.nan
            
    df_train.to_csv(filename, mode='w', index=False, header=True)

if __name__ == "__main__":
    con = psycopg2.connect(database='codeforces', user='Joy')
    cur = con.cursor()

    # note this is 4x faster than getting it from sql
    df_smooth = pd.read_csv('user_ratings_smoothed.csv', engine = 'c')

    cur.execute("SELECT * FROM handles")
    all_handles = [h[0] for h in cur.fetchall()]

    with open('handle_cid_pid_keys.txt') as f:
        keys = [line.strip() for line in f.readlines()]

    print "getting categorical variables"
    binvars = get_categorical_variables([
        'all_participanttypes',
        'all_tags',
        'all_verdicts',
        'all_language'
    ], cur)

    print "reading dfs from sql"
    # ------------------------------------------
    # problem stats
    # ------------------------------------------
    # problem rating and tags
    df_prate = pd.read_sql("SELECT * FROM problem_rating", con)
    df_prate.set_index(['contestid', 'problemid'], inplace=True)

    df_tags = pd.read_sql("SELECT * FROM tags", con)
    df_tags.set_index(['contestid', 'problemid'], inplace=True)
    df_tags.sort_index(inplace=True)

    df_smooth.reset_index(inplace=True)
    df_smooth.set_index(['handle'], inplace=True)
    df_smooth.drop('contestname', axis=1, inplace=True)
    df_smooth.drop('time', axis=1, inplace=True)


    print "creating user dict"
    # ------------------------------------------
    # dictionary of [user] -> [contestid, problemid]
    user_dict = defaultdict(list)
    keys = [k.split(',') for k in keys]
    for k in keys:
        user_dict[ k[0] ].append(k)


    present_handles = set(df_smooth.index)


    cnt = 0


    lastidx = 57
    user = 'chenmark'
    user_rating = df_smooth.loc[user, :]
    getTraining(user, user_rating, df_prate, user_dict[user], binvars, con, df_tags)
#    for i, user in enumerate(all_handles[lastidx:]):
#        if user in present_handles:
#            print lastidx + i, user
#            user_rating = df_smooth.loc[user, :]
#            getTraining(user, user_rating, df_prate, user_dict[user], binvars, con)
#
