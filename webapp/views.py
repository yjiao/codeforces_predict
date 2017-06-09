from flask import render_template
from webapp import app

import json
import plotly

import numpy as np
import pandas as pd

import psycopg2
from flask import request
from a_Model import ModelIt
from ui_functions import smooth_ratings

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
    print df[:10]
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

    df = pd.read_sql_query(query,con)
    df.sort_values(q1, inplace=True)
    df[q1] = pd.to_datetime(df[q1], unit='s')
    print df.info()
    smoothed = smooth_ratings(df)

    rating_history = df.to_dict(orient='records')

    the_result = ''
    the_result = ModelIt(handle,rating_history)

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
            INNER JOIN problem_info
                ON 
                    problem_info.contestid = problem_rating.contestid
                    AND
                    problem_info.problemid = problem_rating.problemid
        """ % handle
    problems = pd.read_sql_query(query,con)
    problems['starttimeseconds'] = pd.to_datetime(problems['starttimeseconds'], unit='s')

    problem_annotation = problems.contestname.str.cat(problems.problemid, sep=' ')

    # -----------------------------------------------------------------------------
    graphs = [
        dict(
            data=[
                dict(
                    x=problems['starttimeseconds'],
                    y=problems['problemrating'],
                    mode='markers',
                    text=problem_annotation,
                    marker=dict(
                        color='rgba(231, 76, 60, .5)',
                        size=5,
                        ),
                    type='scatter',
                    name="problems solved"
                ),
                dict(
                    x=df[q1],
                    y=df[q3],
                    mode='markers',
                    text=df.contestname,
                    marker=dict(
                        color='rgba((52, 152, 219, .5)',
                        size=10,
                        ),
                    type='scatter',
                    name="contest performance"
                ),
                dict(
                    x=df[q1],
                    y=smoothed,
                    type='line',
                    line=dict(
                        color='rgb(0,0,0)',
                        size=10
                        ),
                    name="smoothed user rating",
                ),
            ],
            layout=dict(
                #title='User Rating',
                yaxis=dict(
                    range=[min(1000, min(df.newrating)-100), max(2000, max(problems.problemrating)+100)]
                ),
                plot_bgcolor='rgb(255,255,255)',
                paper_bgcolor='rgb(255,255,255)',
		margin=dict(
			l=50,
			r=50,
			b=100,
			t=0,
			pad=0
		    ),
            )
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # -----------------------------------------------------------------------------


    return render_template("output.html",
            handle = handle,
            keys = [q1, q2, q3],
            rating_history = rating_history[0:10],
            the_result = the_result,
            ids=ids,
            graphJSON=graphJSON,
            title=title
            )
