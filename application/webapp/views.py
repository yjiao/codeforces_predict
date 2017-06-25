import datetime
from flask import render_template
from webapp import app
import pickle

import numpy as np
import pandas as pd

import psycopg2
from flask import request
from ui_functions import *
import xgboost as xgb

@app.route('/')
@app.route('/index')
def index():
    title = "Code Coach"
    return render_template("index.html",
       title = title
       )

@app.route('/output')
def genprofile_output():

    # ------------------------------------------------------------
    # import some information that will be used for every query
    user = 'Joy'
    host = 'localhost'
    dbname = 'codeforces'
    con = None
    con = psycopg2.connect(database = dbname, user = user)

    title = "Code Coach"

    with open("models/xgb_lite.pickle") as f:
	model = pickle.load(f)




    cur = con.cursor()
    #pull 'handle' from input field and store it
    handle = request.args.get('handle')

    if handle==None:
        query = """
        SELECT handle FROM xgb_last
        ORDER BY RANDOM()
        LIMIT 1;
        """

        cur.execute(query);
        handle = cur.fetchall()[0][0]

        print "=================================================="
        print handle
        print "=================================================="


#    # Get user profile image
#    url = 'http://codeforces.com/api/user.info?handles=' + handle
#    print url
#    user_img = requests.get(url).json()['result'][0]['titlePhoto']

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
                    submissions.handle = '%s'
                    AND
                    submissions.verdict = 'OK'
                    AND
                    problem_info.contestid = submissions.contestid
                    AND
                    problem_info.problemid = submissions.problemid
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

    now = pd.to_datetime(datetime.date.today())

    ddays = now - problem_rating.starttimeseconds
    ddays = np.array([d.days  <= 60 for d in ddays])
    user_stats = dict(
        nproblems = len(problem_rating),
        nproblems30 = sum(ddays)
    )


    usr_rate = user_rating.loc[user_rating.index[-1], 'smoothed_3months']

    # -----------------------------------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------------------------------
    xgb_x_df = pd.read_sql("select * from xgb_last where handle='%s'" % handle, con)
    if len(xgb_x_df) == 0:
        delta_str = "Unknown"
    else:
        best_diff, best_diff_str, delta_str, good_tags, bad_tags = predict(model, xgb_x_df)

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
    problem_thresh = float(user_rating.newrating.values[-1]) + best_diff - 50
    problem_inc = 500
    #print "problem thresh:", problem_thresh, "-------------------------------"
    problem_suggest_dict = filter_problems(handle, problem_thresh, problem_inc, highgrowth_users)
    if len(good_tags) > 3:
        good_tags_names = [x[0] for x in good_tags]
        problem_suggest_dict = [p for p in problem_suggest_dict if p['tag'] in good_tags_names]
    present_tags = set([p['tag'] for p in problem_suggest_dict])


    # -----------------------------------------------------------------------------
    # GENERATE PLOT
    # -----------------------------------------------------------------------------
    ids, graphJSON = plot_user_rating(user_rating, problem_rating, problems_solved, handle, ncontests, tags, highgrowth_users)

    #return render_template("output.html",
    return render_template("keen_index.html",
            title=title,
            handle = handle,
            problems = problem_suggest_dict,
            ids=ids,
            graphJSON=graphJSON,
            delta=delta_str,
            good_tags=good_tags,
            bad_tags=bad_tags,
	    unique_tags=sorted(present_tags),
#            user_img = user_img,
            user_stats = user_stats,
            best_diff_str = best_diff_str
            )
