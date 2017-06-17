from flask import render_template
from webapp import app
import pickle

import numpy as np
import pandas as pd

import psycopg2
from flask import request
from a_Model import ModelIt
from ui_functions import *

# ------------------------------------------------------------
# import some information that will be used for every query
user = 'Joy' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'codeforces'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

title = "AI Algorithms Teacher"

with open("models/ols_models_final.pickle") as f:
    models = pickle.load(f)

with open("models/ols_xmax.pickle") as f:
    xmax = pickle.load(f)

last_data = pd.read_csv('models/ols_last.csv')
binshi = models['bins']
binslo = np.roll(binshi, 1)
binshi = binshi[1:]
binslo = binslo[1:]


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = title
       )

@app.route('/output')
def genprofile_output():
    #pull 'handle' from input field and store it
    handle = request.args.get('handle')

    q1 = 'ratingupdatetimeseconds'
    q2 = 'oldrating'
    q3 = 'newrating'

    #query = "SELECT %s, %s, %s FROM user_rating WHERE handle='%s'" % (q1, q2, q3, handle)
    query = "SELECT * FROM user_rating_smooth WHERE handle='%s'" % handle

    user_rating = pd.read_sql_query(query,con)
    user_rating.sort_values(q1, inplace=True)
    user_rating[q1] = pd.to_datetime(user_rating[q1], unit='s')

    rating_history = user_rating.to_dict(orient='records')

    query = """
        SELECT 
            submissions.starttimeseconds,
            submissions.participanttype,
            problem_info.contestid,
            problem_info.problemid,
            problem_info.contestname
        FROM submissions 
            INNER JOIN problem_info
                ON 
                    problem_info.contestid = submissions.contestid
                    AND
                    problem_info.problemid = submissions.problemid
                    AND
                    submissions.handle = '%s'
        """ % handle
    problem_rating = pd.read_sql_query(query,con)
    problem_rating['starttimeseconds'] = pd.to_datetime(problem_rating['starttimeseconds'], unit='s')

    query = """
        SELECT 
            submissions.contestid,
            submissions.problemid,
            probability_solve.solve_probability,
            probability_solve.problem_rating
        FROM submissions 
            INNER JOIN probability_solve 
                ON 
                    submissions.handle = '%s'
                    AND
                    probability_solve.handle = '%s'
                    AND
                    submissions.contestid = probability_solve.contestid
                    AND
                    submissions.problemid = probability_solve.problemid
        """ % (handle, handle)
    df_prob = pd.read_sql_query(query,con)

    problem_rating = pd.merge(df_prob, problem_rating, on=['contestid', 'problemid'])
    problem_rating.drop_duplicates(inplace=True)

    usr_rate = user_rating.loc[user_rating.index[-1], 'smoothed_3months']

    # -----------------------------------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------------------------------
    with open('models/qbins.txt') as f:
        qbins = np.array(map(float, f.readline().split(',')))
    qbins = [round(q, 2) for q in qbins]

    query = """
        SELECT * FROM train_ols_last WHERE handle='%s'
        """ % handle
    ols_x = pd.read_sql_query(query,con)

    usr_data = last_data.loc[handle]
    usr_bin = np.where(
            np.logical_and(binslo <= usr_rate, binshi > usr_rate)
            )[0][0]
    model, delta, good_tags, bad_tags = predict(models, usr_rate, usr_data, usr_bin, xmax)
    print model

    #model, delta = predict(ols_x, qbins)
    print delta, "------------------------------------"
    if model is None:
        delta_str = "Unknown"
    else:
        if delta > 0:
            delta_str = '+'
        if delta < 0:
            delta_str = ''
        delta_str += str(round(delta, 1))

    # -------------------------------------------------------
    # COLLABORATIVE FILTERING DATA FRAMES
    # -------------------------------------------------------

    # problems solved
    delta_r = 100
    query = """
    SELECT handle, count(*) FROM probability_solve
        WHERE smoothed_3months >= %d
        AND smoothed_3months <= %d
    GROUP BY handle;
    """ % (usr_rate-delta_r, usr_rate+delta_r)
    problems_solved = pd.read_sql(query, con)

    # users who were at this rating for at least 5 contests
    query = """
        SELECT handle, COUNT(*) FROM user_rating_smooth
            WHERE
                smoothed_3months >= %d
                AND
                smoothed_3months <= %d
        GROUP BY handle
        HAVING COUNT(*) >= 5
    """ % (usr_rate-delta_r, usr_rate+delta_r)
    cur = con.cursor()
    cur.execute(query)
    all_similar_users = cur.fetchall()
    all_similar_users = set([x[0] for x in all_similar_users])

    # users who was at this point and then got 100 points better
    query = """
        SELECT handle, COUNT(*) FROM user_rating_smooth
            WHERE
                smoothed_3months >= %d
        GROUP BY handle
        HAVING COUNT(*) >= 5
    """ % (usr_rate+100)
    cur = con.cursor()
    cur.execute(query)
    highgrowth_users = cur.fetchall()
    highgrowth_users = set([x[0] for x in highgrowth_users])
    highgrowth_users = all_similar_users.intersection(highgrowth_users)

    query = """
    SELECT handle, count(*) FROM user_rating_smooth
        WHERE smoothed_3months >= %d
        AND smoothed_3months <= %d
    GROUP BY handle;
    """ % (usr_rate-delta_r, usr_rate+delta_r)
    ncontests = pd.read_sql(query, con)


    query = """
    SELECT submissions.contestid, submissions.problemid, tags.tag
    FROM submissions
	INNER JOIN tags
	ON
	submissions.handle = '%s'
	AND
	submissions.contestid = tags.contestid
	AND
	submissions.problemid = tags.problemid
    """ % (handle)
    tags = pd.read_sql(query, con)
    tags.drop_duplicates(inplace=True)
    tags = tags.tag.values

    # -----------------------------------------------------------------------------
    # PROBLEM SUGGESTIONS
    # -----------------------------------------------------------------------------
    problem_thresh = float(user_rating.newrating.values[-1]) + 300
    problem_inc = 300
    print problem_thresh, "-------------------------------"
    problem_suggest_dict = filter_problems(handle, problem_thresh, problem_inc, highgrowth_users)

    # -----------------------------------------------------------------------------
    # GENERATE PLOT
    # -----------------------------------------------------------------------------
    ids, graphJSON = plot_user_rating(user_rating, problem_rating, delta, model, problems_solved, handle, ncontests, tags, highgrowth_users)

    #return render_template("output.html",
    return render_template("keen_index.html",
            title=title,
            handle = handle,
            problems = problem_suggest_dict,
            ids=ids,
            graphJSON=graphJSON,
            delta=delta_str,
            good_tags=good_tags,
            bad_tags=bad_tags
            )
