# colors: https://color.adobe.com/Flat-UI-color-theme-2469224/edit/?copy=true&base=1&rule=Custom&selected=4&name=Copy%20of%20Flat%20UI&mode=rgb&rgbvalues=0.172549,0.243137,0.313725,0.905882,0.298039,0.235294,0.92549,0.941176,0.945098,0.203922,0.596078,0.858824,0.160784,0.501961,0.72549&swatchOrder=0,1,2,3,4
import scipy
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import psycopg2
import tensorflow as ts
from collections import defaultdict

import json
import plotly

import statsmodels.api as sm

import datetime

con = psycopg2.connect(database='codeforces', user='Joy')
cur = con.cursor()

def smooth_ratings(df):

    window = df.shape[0]/3
    window += ((window%2) == 0)

    smooth_rating = scipy.signal.savgol_filter(df.newrating.values, window, polyorder=1, axis=0)

    return smooth_rating


def plot_user_rating(user_rating, problems, delta, model):
    now = pd.to_datetime(datetime.date.today())

    smoothed = smooth_ratings(user_rating)
    problem_annotation = problems.contestname.str.cat(problems.problemid, sep=' ')

    # predictions
    x_pred =[now]
    last_rating = user_rating.newrating.values[-1]
    y_pred =[last_rating + delta]

    # for drawing squares
    x0 = min(problems.starttimeseconds)
    x1 = now

    colors = {
            'gray'  : 'rgba(222, 222, 222 .2)',
            'green' : 'rgba(175, 227, 127, .2)',
            'teal'  : 'rgba(127, 227, 225, .2)',
            'blue'  : 'rgba(78, 155, 237, .2)',
            'purple': 'rgba(222, 108, 235, .2)',
            'yellow': 'rgba(237, 205, 78, .2)',
            'red'   : 'rgba(237, 121, 121, .2)'
            }
    cutoffs = {
            'gray'  : [0, 1199],
            'green' : [1200, 1399],
            'teal'  : [1400, 1599],
            'blue'  : [1600, 1899],
            'purple': [1900, 2199],
            'yellow': [2200, 2399],
            'red'   : [2400, 5000]
            }

    # -----------------------------------------------------------------------------
    shapes=[
        { 'type': 'rect', 'x0': x0, 'y0': cutoffs['gray'][0], 'x1': x1, 'y1': cutoffs['gray'][1], 'line': { 'color': 'rgba(128, 0, 128, 0)', 'width': 0, }, 'fillcolor': colors['gray']},
        { 'type': 'rect', 'x0': x0, 'y0': cutoffs['green'][0], 'x1': x1, 'y1': cutoffs['green'][1], 'line': { 'color': 'rgba(128, 0, 128, 0)', 'width': 0, }, 'fillcolor': colors['green']},
        { 'type': 'rect', 'x0': x0, 'y0': cutoffs['teal'][0], 'x1': x1, 'y1': cutoffs['teal'][1], 'line': { 'color': 'rgba(128, 0, 128, 0)', 'width': 0, }, 'fillcolor': colors['teal']},
        { 'type': 'rect', 'x0': x0, 'y0': cutoffs['blue'][0], 'x1': x1, 'y1': cutoffs['blue'][1], 'line': { 'color': 'rgba(128, 0, 128, 0)', 'width': 0, }, 'fillcolor': colors['blue']},
        { 'type': 'rect', 'x0': x0, 'y0': cutoffs['purple'][0], 'x1': x1, 'y1': cutoffs['purple'][1], 'line': { 'color': 'rgba(128, 0, 128, 0)', 'width': 0, }, 'fillcolor': colors['purple']},
        { 'type': 'rect', 'x0': x0, 'y0': cutoffs['yellow'][0], 'x1': x1, 'y1': cutoffs['yellow'][1], 'line': { 'color': 'rgba(128, 0, 128, 0)', 'width': 0, }, 'fillcolor': colors['yellow']},
        { 'type': 'rect', 'x0': x0, 'y0': cutoffs['red'][0], 'x1': x1, 'y1': cutoffs['red'][1], 'line': { 'color': 'rgba(128, 0, 128, 0)', 'width': 0, }, 'fillcolor': colors['red']}
        ]

    # -----------------------------------------------------------------------------
    trace_model = dict(
        y=['Mean of (problem difficulty) - (user level)',
            '#problems where (problem difficulty) > (user level)',
            '# wrong tries/ problem',
            'Std of (problem difficulty) - (user level)',
            '#contests in this period',
            'Your current (smoothed) rating',
            'Previous change'],
        x=model.params,
        marker=dict(
            color='rgba(231, 76, 60, 1)',
            size=5,
            ),
        type='bar',
        name="Model parameters",
        orientation = 'h'
    )

    trace_problemssolved = dict(
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
    )

    trace_userrating = dict(
        x=user_rating.ratingupdatetimeseconds,
        y=user_rating.newrating,
        mode='markers',
        text=user_rating.contestname,
        marker=dict(
            color='rgba((52, 152, 219, .5)',
            size=10,
            ),
        type='scatter',
        name="contest performance"
    )

    trace_smooth = dict(
        x=user_rating.ratingupdatetimeseconds,
        y=smoothed,
        type='line',
        line=dict(
            color='rgb(0,0,0)',
            size=10
            ),
        name="smoothed user rating",
    )

    trace_prediction = dict(
        x=x_pred,
        y=y_pred,
        marker=dict(
            color='rgba(0, 0, 0, 1)',
            size=10,
            ),
        mode='markers',
        type='scatter',
        name="prediction"
    )

    # -----------------------------------------------------------------------------
    model_plot = dict(
        data=[trace_model],
        layout=dict(
            autosize = "False",
            height=200,
            plot_bgcolor='rgb(255,255,255)',
            paper_bgcolor='rgb(255,255,255)',
            margin=dict(
                    l=350,
                    r=200,
                    b=50,
                    t=0,
                    pad=0
                )
            )
    )

    rating_plot = dict(
        data=[ trace_problemssolved, trace_userrating, trace_smooth, trace_prediction ],
        layout=dict(
            yaxis=dict(
                range=[min(1000, min(user_rating.newrating)-100), max(2000, max(problems.problemrating)+100)]
            ),
            autosize = "False",
            plot_bgcolor='rgb(255,255,255)',
            paper_bgcolor='rgb(255,255,255)',
            margin=dict(
                    l=50,
                    r=50,
                    b=50,
                    t=0,
                    pad=0
                ),
            shapes=shapes,
            hovermode='closest'
        )
    )


    # -----------------------------------------------------------------------------
    graphs = [
        rating_plot,
        model_plot
    ]

    # Add "ids" to each of the graphs to pass up to the client for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON


def predict(x, qbins):
    xbin = round(x['bin'].values[0], 2)
    bin_idx = qbins.index(xbin)
    
    model_file = "models/ols%d.pickle" % bin_idx
    print "USING MODEL", model_file, "----------------------------"
    model = sm.load(model_file)

    cols = ['rating_diff_mean',
            'n_harder',
            'n_wrong_mean',
            'rating_diff_std',
            'n_contest',
            'smoothed_3months',
            'prev']

    future_val = model.predict(x[cols])[0]
    return model, future_val

def filter_problems(problem_thresh, problem_inc):
    query = """
        SELECT 
            problem_rating.problemrating,
            problem_info.contestid,
            problem_info.problemid,
            problem_info.contestname,
            problem_info.name,
            problem_info.division,
            problem_info.points
        FROM problem_info
            INNER JOIN problem_rating
                ON 
                    problem_info.contestid = problem_rating.contestid
                    AND
                    problem_info.problemid = problem_rating.problemid
                    AND
                    problem_rating.problemrating >= %d
                    AND
                    problem_rating.problemrating <= %d
        ORDER BY problem_rating.contestid DESC
        LIMIT 20
        """ % (problem_thresh, problem_thresh + problem_inc)
    problem_suggest = pd.read_sql_query(query,con)
    return problem_suggest.to_dict(orient='records')








