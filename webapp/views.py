from flask import render_template
from webapp import app

import numpy as np
import pandas as pd

import psycopg2
from flask import request
from a_Model import ModelIt
from ui_functions import *

user = 'Joy' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'codeforces'
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

title = "AI Algorithms Teacher"

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = title
       )

@app.route('/db')
def birth_page():
    sql_query = """                                                             
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
;                                                                               
                """
    df = pd.read_sql_query(sql_query,con)
    rating_history = ""
    for i in range(0,10):
        rating_history += df.iloc[i]['handle']
        rating_history += "<br>"
    return rating_history

@app.route('/db_fancy')
def genprofile_page_fancy():
    sql_query = """
               SELECT index, attendant, handle FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    df=pd.read_sql_query(sql_query,con)
    rating_history = []
    for i in range(0,df.shape[0]):
        rating_history.append(dict(index=df.iloc[i]['index'], attendant=df.iloc[i]['attendant'], handle=df.iloc[i]['handle']))
    return render_template('genprofile.html',rating_history=rating_history)

@app.route('/input')
def genprofile_input():
    return render_template("input.html",
            title=title)

@app.route('/output')
def genprofile_output():
    #pull 'handle' from input field and store it
    handle = request.args.get('handle')

    q1 = 'ratingupdatetimeseconds'
    q2 = 'oldrating'
    q3 = 'newrating'

    #query = "SELECT %s, %s, %s FROM user_rating WHERE handle='%s'" % (q1, q2, q3, handle)
    query = "SELECT * FROM user_rating WHERE handle='%s'" % handle

    user_rating = pd.read_sql_query(query,con)
    user_rating.sort_values(q1, inplace=True)
    user_rating[q1] = pd.to_datetime(user_rating[q1], unit='s')

    rating_history = user_rating.to_dict(orient='records')

    query = """
        SELECT 
            submissions.starttimeseconds,
            submissions.participanttype,
            problem_rating.problemrating,
            problem_info.contestid,
            problem_info.problemid,
            problem_info.contestname
        FROM submissions 
            INNER JOIN problem_rating
                ON 
                    submissions.contestid = problem_rating.contestid
                    AND
                    submissions.problemid = problem_rating.problemid
                    AND
                    submissions.handle = '%s'
                    AND
                    submissions.verdict = 'OK'
            INNER JOIN problem_info
                ON 
                    problem_info.contestid = problem_rating.contestid
                    AND
                    problem_info.problemid = problem_rating.problemid
        """ % handle
    problem_rating = pd.read_sql_query(query,con)
    problem_rating['starttimeseconds'] = pd.to_datetime(problem_rating['starttimeseconds'], unit='s')

    # -----------------------------------------------------------------------------
    # PROBLEM SUGGESTIONS
    problem_thresh = float(user_rating.newrating.values[-1]) + 300
    problem_inc = 300
    print problem_thresh, "-------------------------------"
    problem_suggest_dict = filter_problems(problem_thresh, problem_inc)

    # -----------------------------------------------------------------------------
    # MODEL PREDICTION
    with open('models/qbins.txt') as f:
        qbins = np.array(map(float, f.readline().split(',')))
    qbins = [round(q, 2) for q in qbins]

    query = """
        SELECT * FROM train_ols_last WHERE handle='%s'
        """ % handle
    ols_x = pd.read_sql_query(query,con)

    model, delta = predict(ols_x, qbins)
    print delta, "------------------------------------"
    if delta > 0:
        delta_str = '+'
    if delta < 0:
        delta_str = ''
    delta_str += str(round(delta, 1))

    # -----------------------------------------------------------------------------
    # Generate plot
    ids, graphJSON = plot_user_rating(user_rating, problem_rating, delta, model)

    return render_template("output.html",
            title=title,
            handle = handle,
            problems = problem_suggest_dict,
            ids=ids,
            graphJSON=graphJSON,
            delta=delta_str
            )

