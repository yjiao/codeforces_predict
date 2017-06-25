# feature engineering for ensemble regression
# uses unsmoothed data
# parallel computing per user, for running on google compute engine

import pandas as pd
import numpy as np
from collections import Counter
from joblib import Parallel, delayed

# ---------------------------------------------
# constants
secinday = 3600.0*24.0 # days

# --------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------

def get_binary_dictionary():
    # ---------------------------------------------
    # some hard-coded groupings of sparse features
    # languages
    java = [ 'Java 6', 'Java 8', 'Java 8 ZIP', 'Java 7' ]
    python = [ 'Python 3', 'Python 2', 'PyPy 2', 'PyPy 3']
    lowlevel = [ 'GNU C++14', 'GNU C', 'MS C++', 'GNU C++', 'GNU C++0x', 'MS C#', 'GNU C11', 'GNU C++11 ZIP', 'GNU C++11', 'Mono C#', 'Delphi', ]
    # verdicts 
    errors = [ "COMPILATION_ERROR", "RUNTIME_ERROR", "CRASHED", "REJECTED", "IDLENESS_LIMIT_EXCEEDED"]
    wrong = [ "TIME_LIMIT_EXCEEDED", "WRONG_ANSWER", "CHALLENGED", "MEMORY_LIMIT_EXCEEDED" ]
    # practice mdoes
    practice = [ 'GYM', 'OUT_OF_COMPETITION', 'VIRTUAL', 'PRACTICE' ]
    bindict = {}

    for j in java:
        bindict[j] = 'java'
    for p in python:
        bindict[p] = 'python'
    for c in lowlevel:
        bindict[c] = 'lowlevel'

    for v in errors:
        bindict[v] = 'error'
    for v in wrong:
        bindict[v] = 'wrong'
    for v in practice:
        bindict[v] = 'practice'

    bindict['OK'] = 'ok'
    bindict['CONTESTANT'] = 'contestant'

    return bindict

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

def prob_solve(user_rating, problem_rating):
    return 1.0 / ( 1 + 10 ** ((problem_rating - user_rating ) / 400.0) )

# --------------------------------------------------------
# MAIN FUNCTIONS
# --------------------------------------------------------
def get_features(dfw, prevcontest, nextcontest, prevtags, tags, prevcontestdf, month=3):
    rating_column = 'smoothed_%dmonths' % month
    #change_column = 'delta_smoothed_%dmonths' % month
    dfw.fillna(value=0, inplace=True)

    # --------------------------------------------------------------------
    # number of wrong problems
    # num problems wrong in practice
    mean_wrong_contest,\
        max_wrong_contest,\
        total_wrong_contest,\
        std_wrong_contest = get_stats_df(dfw, 'wrong', dfw.contestant > 0)
    # num problems wrong in contest
    mean_wrong_practice,\
        max_wrong_practice,\
        total_wrong_practice,\
        std_wrong_practice = get_stats_df(dfw, 'wrong', dfw.practice > 0)

    # --------------------------------------------------------------------
    # number of error
    # num problems error in practice
    mean_error_contest,\
        max_error_contest,\
        total_error_contest,\
        std_error_contest = get_stats_df(dfw, 'error', dfw.contestant > 0)
    # num problems error in contest
    mean_error_practice,\
        max_error_practice,\
        total_error_practice,\
        std_error_practice = get_stats_df(dfw, 'error', dfw.practice > 0)

    # --------------------------------------------------------------------
    # difference between user rating and problem rating
    dfw['ratingdiff'] = dfw['problem_rating'] - prevcontest[rating_column]
    mean_ratingdiff_contest,\
            max_ratingdiff_contest, _,\
            std_ratingdiff_contest = get_stats_df(dfw, 'ratingdiff', dfw.contestant > 0)
    mean_ratingdiff_practice,\
            max_ratingdiff_practice, _,\
            std_ratingdiff_practice = get_stats_df(dfw, 'ratingdiff', dfw.practice > 0)

    # --------------------------------------------------------------------
    # probability of solving question
    dfw['probability'] = dfw['problem_rating'].apply( lambda x: prob_solve(prevcontest[rating_column], x )  )
    mean_probability_contest,\
            max_probability_contest,\
            min_probability_contest,\
            std_probability_contest = get_stats_df(dfw, 'probability', dfw.contestant > 0)
    mean_probability_practice,\
            max_probability_practice,\
            min_probability_practice,\
            std_probability_practice = get_stats_df(dfw, 'probability', dfw.practice > 0)

    # --------------------------------------------------------------------
    # unlikely solves
    n_unlikely02 = sum(dfw.probability <= 0.2)
    n_unlikely01 = sum(dfw.probability <= 0.1)

    # --------------------------------------------------------------------
    # probability of performance on contest
    g = prevcontestdf.groupby(['contestid', 'problemid'])
    performance = 1.0
    usr_rating = prevcontest[rating_column]
    for k, v in g:
        cid = k[0]
        pid = k[1]
        pr = dfw.loc[ 
                np.logical_and(
                    dfw.contestid == cid,
                    dfw.problemid == pid
                    ), 'problem_rating'
                ].values[0]
        if pr == -1:
            continue
        if "OK" in v.verdict.values:
            performance *= prob_solve(usr_rating, pr)
        else:
            ptmp = (1 - prob_solve(usr_rating, pr))
            performance *= ptmp

    # --------------------------------------------------------------------
    # Time between solves
    timediff = dfw.starttimeseconds - np.roll(dfw.starttimeseconds, 1)
    timediff = timediff[1:] / 3600.0 / 24.0
    # (mean, max_, min_, tot, std)
    mean_timediff, max_timediff, min_timediff, _, std_timediff = get_stats_col(timediff)

    # --------------------------------------------------------------------
    # Time between first submit and solves
    # (mean, max_, min_, tot, std)
    mean_solvetime,\
            max_solvetime,\
            min_solvetime,\
            tot_solvetime,\
            std_solvetime = get_stats_col(dfw.hours_submit_to_solve)

    # --------------------------------------------------------------------
    # num problems solved > threshold
    # (mean, max_, min_, tot, std)
    if 'solvetimeseconds' not in dfw.columns:
        n100 = 0
        n200 = 0
        n300 = 0
        n400 = 0
        n500 = 0
        n_solved = 0
    else:
        _, _, _, n100, _ = get_stats_col(np.logical_and(dfw.ratingdiff >= 100, dfw.solvetimeseconds > 0))
        _, _, _, n200, _ = get_stats_col(np.logical_and(dfw.ratingdiff >= 200, dfw.solvetimeseconds > 0))
        _, _, _, n300, _ = get_stats_col(np.logical_and(dfw.ratingdiff >= 300, dfw.solvetimeseconds > 0))
        _, _, _, n400, _ = get_stats_col(np.logical_and(dfw.ratingdiff >= 400, dfw.solvetimeseconds > 0))
        _, _, _, n500, _ = get_stats_col(np.logical_and(dfw.ratingdiff >= 500, dfw.solvetimeseconds > 0))
        n_solved = sum(dfw.solvetimeseconds > 0)

    # --------------------------------------------------------------------
    # tags
    # number of new tags
    newtags = set(tags.keys()).difference(prevtags.keys())
    total_newtags = 0
    unique_newtags = len(newtags)
    for nt in newtags:
        total_newtags += tags[nt]

    alltags = prevtags.keys()
    alltags.extend(tags.keys())
    unique_tags_total = len(set( alltags ))


    # --------------------------------------------------------------------
    # rate of change
#    dr = prevcontest[change_column]
#    dt = max(dfw.starttimeseconds) - prevcontest.starttimeseconds
#    drdt = (dr + 0.0) / dt

    # --------------------------------------------------------------------
    # languages
    langcounts = np.sum(dfw[['java', 'python', 'lowlevel']]) > 0

    # --------------------------------------------------------------------
    features = {
	"n_solved"                 :    n_solved,
	"mean_wrong_contest"       :    mean_wrong_contest,
	"max_wrong_contest"        :    max_wrong_contest,
	"total_wrong_contest"      :    total_wrong_contest,
	"std_wrong_contest"        :    std_wrong_contest,
	"mean_wrong_practice"      :    mean_wrong_practice,
	"max_wrong_practice"       :    max_wrong_practice,
	"total_wrong_practice"     :    total_wrong_practice,
	"std_wrong_practice"       :    std_wrong_practice,
	"mean_error_contest"       :    mean_error_contest,
	"max_error_contest"        :    max_error_contest,
	"total_error_contest"      :    total_error_contest,
	"std_error_contest"        :    std_error_contest,
	"mean_error_practice"      :    mean_error_practice,
	"max_error_practice"       :    max_error_practice,
	"total_error_practice"     :    total_error_practice,
	"std_error_practice"       :    std_error_practice,
	"mean_ratingdiff_contest"  :    mean_ratingdiff_contest,
	"max_ratingdiff_contest"   :    max_ratingdiff_contest,
	"std_ratingdiff_contest "  :    std_ratingdiff_contest,
	"mean_ratingdiff_practice" :    mean_ratingdiff_practice,
	"max_ratingdiff_practice"  :    max_ratingdiff_practice,
	"std_ratingdiff_practice"  :    std_ratingdiff_practice,
	"mean_timediff"            :    mean_timediff,
	"max_timediff"             :    max_timediff,
	"min_timediff"             :    min_timediff,
	"std_timediff"             :    std_timediff,
	"mean_solvetime"           :    mean_solvetime,
	"max_solvetime"            :    max_solvetime,
	"min_solvetime"            :    min_solvetime,
	"tot_solvetime"            :    tot_solvetime,
	"std_solvetime"            :    std_solvetime,
        "mean_probability_contest" :    mean_probability_contest,
        "max_probability_contest"  :    max_probability_contest,
        "min_probability_contest"  :    min_probability_contest,
        "std_probability_contest"  :    std_probability_contest,
        "mean_probability_practice":    mean_probability_practice,
        "max_probability_practice" :    max_probability_practice,
        "min_probability_practice" :    min_probability_practice,
        "std_probability_practice" :    std_probability_practice,
	"n100"                     :    n100,
	"n200"                     :    n200,
	"n300"                     :    n300,
	"n400"                     :    n400,
	"n500"                     :    n500,
        "unique_newtags"           :    unique_newtags,
        "total_newtags"            :    total_newtags,
        "unique_tags_total"        :    unique_tags_total,
        'performance'              :    performance,
        'java'                     :    int(langcounts['java']),
        'python'                   :    int(langcounts['python']),
        'lowlevel'                 :    int(langcounts['lowlevel']),
        'n_unlikely01'             :    n_unlikely01,
        'n_unlikely02'             :    n_unlikely02,
#	"drdt"                     :    drdt
	}

    return features

def get_df_problem(df, subdict, prevcontest, nextcontest, dfrat, df_prate, df_tags, bindict, month=3):
    trainlist = []
    # --------------------------------------------------
    # per-problem features
    gprob = df.groupby(['contestid', 'problemid'])
    for k, v in gprob:
        cid = k[0]
        pid = k[1]
        ex = dict()
        # generic problem info
        ex['points'] = v.points.values[0]
        ex['problemid'] = v.problemid.values[0]
        ex['contestid'] = v.contestid.values[0]

        # ----------------------------------
        # user rating info
        # find closest PREVIOUS contest
        # if there is no next contest,then skip this entry
        ex['starttimeseconds'] = min(v.starttimeseconds)
        ex['stoptimeseconds'] = max(v.starttimeseconds)
        
        # ----------------------------------
        # problem rating and probability of solving
        if (cid, pid) in df_prate.index:
            ex['problem_rating'] = df_prate.loc[cid, pid].values[0]
        else:
            ex['problem_rating'] = -1
            
        # ----------------------------------
        # time to solves
        solvetime = v.loc[v.verdict=='OK', 'starttimeseconds']
        if len(solvetime) > 0:
            ex['solvetimeseconds'] = min(solvetime)
            ex['hours_submit_to_solve'] = (ex['solvetimeseconds'] - ex['starttimeseconds']) / 3600.0
        else:
            # some problems were never solved
            ex['hours_submit_to_solve'] = -1
        
        # -----------------------------------------------------------------------
        # tags
        idx_tag = (cid, pid)
        if idx_tag in df_tags.index:
            for t in df_tags.loc[idx_tag]['tag'].values:
                ex[t] = 1

        # -----------------------------------------------------------------------
        # binary variables that should be grouped
        # languages
        lang = v['language'].values[0]
        if lang in bindict:
            ex[bindict[ lang ]] = 1
        # verdicts
        vcnt = v.verdict.value_counts()
        vdict = vcnt.to_dict()
        for key, val in vdict.iteritems():
            if key in bindict:
                ex[ bindict[ key ] ] = val
        # participant type
        pcnt = v.participanttype.value_counts()
        pdict = pcnt.to_dict()
        for t in pdict.iterkeys():
            if t in bindict:
                ex[ bindict[t] ] = 1

        # add in any missing binary variables
        for bincol in bindict.itervalues():
            if bincol not in ex:
                ex[bincol] = np.nan

        trainlist.append(ex)
    df_problems = pd.DataFrame.from_dict(trainlist)

    return df_problems

def summarize(qfront, qback, subdict, prevcontest, nextcontest, dfrat, df_prate, df_tags, bindict, prevtags, month=3):
    df = pd.DataFrame.from_dict(subdict[qfront:qback+1])

    pcid = dfrat.loc[prevcontest]['contestid']
    #print "PREV:", pcid
    idx = np.logical_and(
            df.participanttype == 'CONTESTANT',
            df.contestid == pcid
            )
    prevcontestdf = df.loc[idx]

    df_problems = get_df_problem(df, subdict, prevcontest, nextcontest, dfrat, df_prate, df_tags, bindict)

    # grab new tags
    tagkey = df_problems[['contestid', 'problemid']].values
    tags = []
    for tk in tagkey:
        for t in df_tags.loc[tk].values:
            tags.append(t[0])
    tags = Counter(tags)
    features = get_features(df_problems, dfrat.loc[prevcontest], dfrat.loc[nextcontest], prevtags, tags, prevcontestdf )

    prevrec = dfrat.loc[prevcontest]
    prevrec = prevrec.to_dict()
    prevrec.pop('index')
    features.update(prevrec)

    return tags, features

def parse_user(dfsub, handle, df_smooth, df_prate, df_tags, bindict):
    # --------------------------
    # per user
    dfsub.is_copy = False
    dfsub['type'] = 'problem'

    dfrat = df_smooth.loc[df_smooth.handle == handle]
    dfrat.is_copy = False

    dfsub.sort_values('starttimeseconds', inplace=True)
    dfrat.sort_values('starttimeseconds', inplace=True)
    dfsub.reset_index(inplace=True)
    dfrat.reset_index(inplace=True)

    dfsub['index'] = dfsub.index
    dfrat['index'] = dfrat.index

    subdict = dfsub.to_dict(orient='records')
    ratdict = dfrat.to_dict(orient='records')

    idx = dfsub[['starttimeseconds', 'type', 'index']]
    idx = idx.to_dict(orient="records")
    idx_rat = dfrat[['starttimeseconds', 'type', 'index']]
    idx_rat = idx_rat.to_dict(orient="records")

    idx.extend(idx_rat)
    idx.sort(key = lambda x: x['starttimeseconds'])

    # note we are using a list as a queue since the python Queue class has heavy overhead
    # including locks, which we do not need
    qfront = 0
    qback = 0
    cutoff = 30
    prevcontest = -1
    prevtags = {}
    trainlist = []
    qempty = False

    # historical totals
    n_contests = 0
    n_problems = 0

    for i in idx:
        if qfront >= len(subdict):
            break
        if i['type'] == 'problem':
            qback += 1
            n_problems += 1
        else:
            assert(i['type'] == 'contest')
            # found contest
            curtime = i['starttimeseconds']
            days_elapsed = (curtime - subdict[qfront]['starttimeseconds']) / secinday
            # remove problems that occurred too long ago
            while days_elapsed > cutoff:
                qfront += 1
                if qfront >= len(subdict):
                    qempty = True
                    break

                days_elapsed = (curtime - subdict[qfront]['starttimeseconds']) / secinday

            if qempty:
                break

            # process previous problems solved
            if prevcontest == -1:
                tags, features = summarize(qfront, qback, subdict, i['index'], i['index'], dfrat, df_prate, df_tags, bindict, prevtags)
            else:
                tags, features = summarize(qfront, qback, subdict, prevcontest, i['index'], dfrat, df_prate, df_tags, bindict, prevtags)

            features['total_contests'] = n_contests
            features['total_problems'] = n_problems

            n_contests += 1
            trainlist.append(features)

            prevtags.update(Counter(tags))
            prevcontest = i['index']
    ret = pd.DataFrame.from_dict(trainlist)
    return ret

if __name__ == "__main__":
    # smoothed user ratings
    df_smooth = pd.read_csv('csv/user_ratings_smoothed.csv', engine = 'c')
    df_smooth['type'] = 'contest'
    df_smooth['starttimeseconds'] = df_smooth['ratingupdatetimeseconds']
    df_smooth.drop('contestname', axis=1, inplace=True)
    df_smooth.drop('time', axis=1, inplace=True)
    present_handles = list(set(df_smooth.handle))

    print "handles: ", len(present_handles)

    # problem ratings
    df_prate = pd.read_csv('csv/problem_ratings.csv', engine='c')
    df_prate.set_index(['contestid', 'problemid'], inplace=True)

    # problem tags
    df_tags= pd.read_csv('csv/all_tags.csv', engine='c')
    df_tags.set_index(['contestid', 'problemid'], inplace=True)
    df_tags.sort_index(inplace=True)
    
    print "reading submissions..."
    # submissions
    df_submissions = pd.read_csv('csv/all_submissions.csv', engine='c')
    df_submissions.columns = [
        'handle',
        'contestid',
        'submissionid',
        'language',
        'memoryBytes',
        'participanttype',
        'passedtestcount',
        'points',
        'problemid',
        'problem_name',
        'problem_tags',
        'relativetimeseconds',
        'starttimeseconds',
        'testset',
        'timemilliseconds',
        'verdict'
    ]
    print "done"


    bindict = get_binary_dictionary()

    lastidx = 0
    stopidx = len(present_handles)

    def processInput(i, user):
        dfsub = df_submissions.loc[df_submissions.handle == user]
        print lastidx + i, user
        user_rating = df_smooth.loc[df_smooth.handle == user]
        data = parse_user(dfsub, user, df_smooth, df_prate, df_tags, bindict)
        data.to_csv("ols_train/%s.csv" % user, index=False)

    Parallel(n_jobs=8)(delayed(processInput)(i, user) for i, user in enumerate(present_handles[lastidx:stopidx]))