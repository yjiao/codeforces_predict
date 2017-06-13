import pandas as pd
import numpy as np
import psycopg2

# ---------------------------------------------
# some hard-coded groupings of sparse features
# languages
java = [ 'Java 6', 'Java 8', 'Java 8 ZIP', 'Java 7' ]
python = [ 'Python 3', 'Python 2', 'PyPy 2', 'PyPy 3']
lowlevel = [ 'GNU C++14', 'GNU C', 'MS C++', 'GNU C++', 'GNU C++0x', 'MS C#', 'GNU C11', 'GNU C++11 ZIP', 'GNU C++11', 'Mono C#', 'Delphi', ]
otherlang = [ 'Haskell', 'Factor', 'Picat', 'Secret_171', 'PHP', 'Tcl', 'Scala', 'Io', 'FPC', 'J', 'Rust', 'JavaScript', 'Ada', 'Go', 'Cobol', 'Befunge', 'R oco', 'Ruby', 'Kotlin', 'F#', 'Perl', 'Pike', 'D', 'Ocaml' ]
# verdicts 
errors = [ "COMPILATION_ERROR", "RUNTIME_ERROR", "CRASHED", "REJECTED", "IDLENESS_LIMIT_EXCEEDED"]
wrong = [ "TIME_LIMIT_EXCEEDED", "WRONG_ANSWER", "CHALLENGED", "MEMORY_LIMIT_EXCEEDED" ]
# practice mdoes
practice = [ 'GYM', 'OUT_OF_COMPETITION', 'VIRTUAL', 'PRACTICE' ]
# columns that are too sparse or reduandant to be informative
drop = [ 'FALSE', 'SKIPPED', 'TESTING', 'PARTIAL', 'REJECTED', 'PRESENTATION_ERROR', 'FAILED', 'rank' ]

# ---------------------------------------------
# constants
secinday = 3600.0*24.0 # days

def get_categorical_variables( colnames, cur):                                                                                                              
    catvars = []
    for c in colnames:
        cur.execute("select * from %s" % c, con)
        catvars.extend([c[1] for c in cur.fetchall()])

    return catvars

def summarize(qfront, qback, subdict, prevcontest, nextcontest, dfrat, df_prate, df_tags, month=3):
    df = pd.DataFrame.from_dict(subdict[qfront:qback+1])
    
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
        # verdicts
        vcnt = v.verdict.value_counts()
        vdict = vcnt.to_dict()
        ex.update(vdict)

        # ----------------------------------
        # participant type
        pcnt = v.participanttype.value_counts()
        pdict = pcnt.to_dict()
        for t in pdict.iterkeys():
            ex[t] = 1
            
        # ----------------------------------
        # problem rating
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
        if (idx_tag) in df_tags:
            for t in df_tags.loc[idx_tag]['tag'].values:
                ex[t] = 1
    
        print ex

    smoothratinglabel = "smoothed_%dmonths" % month
    print dfrat.loc[prevcontest]['contestid'], dfrat.loc[nextcontest]['contestid']

def parse_user(handle, con, df_smooth, df_prate, df_tags):
    # --------------------------
    # per user
    dfsub = pd.read_sql("select * from submissions where handle='%s'" % handle, con)
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

    for i in idx:
        if i['type'] == 'problem':
            qback += 1
        else:
            assert(i['type'] == 'contest')
            # found contest
            curtime = i['starttimeseconds']
            days_elapsed = (curtime - subdict[qfront]['starttimeseconds']) / secinday
            # remove problems that occurred too long ago
            while days_elapsed > cutoff:
                qfront += 1
                days_elapsed = (curtime - subdict[qfront]['starttimeseconds']) / secinday
            # process previous problems solved
            #print ratdict[i['index']]['contestid'], "-------------"
            if prevcontest == -1:
                summarize(qfront, qback, subdict, i['index'], i['index'], dfrat, df_prate, df_tags)
            else:
                summarize(qfront, qback, subdict, prevcontest, i['index'], dfrat, df_prate, df_tags)
            prevcontest = i['index']

if __name__ == "__main__":
    con = psycopg2.connect(database='codeforces', user='Joy')

    # smoothed user ratings
    df_smooth = pd.read_csv('user_ratings_smoothed.csv', engine = 'c')
    df_smooth['type'] = 'contest'
    df_smooth['starttimeseconds'] = df_smooth['ratingupdatetimeseconds']
    df_smooth.drop('contestname', axis=1, inplace=True)
    df_smooth.drop('time', axis=1, inplace=True)

    # problem ratings
    df_prate = pd.read_sql("SELECT * FROM problem_rating", con)                                                                                                 
    df_prate.set_index(['contestid', 'problemid'], inplace=True)

    # problem tags
    df_tags = pd.read_sql("SELECT * FROM tags", con)
    df_tags.set_index(['contestid', 'problemid'], inplace=True)
    df_tags.sort_index(inplace=True)

    cur.execute("SELECT * FROM handles")
    all_handles = [h[0] for h in cur.fetchall()]
    present_handles = set(df_smooth.index)

    lastidx = 0
    for i, user in enumerate(all_handles[lastidx:]):
        if user in present_handles:
            print lastidx + i, user
            user_rating = df_smooth.loc[user, :]
            parse_user(user, con, df_tags)



