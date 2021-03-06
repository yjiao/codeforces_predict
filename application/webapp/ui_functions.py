# colors: https://color.adobe.com/Flat-UI-color-theme-2469224/edit/?copy=true&base=1&rule=Custom&selected=4&name=Copy%20of%20Flat%20UI&mode=rgb&rgbvalues=0.172549,0.243137,0.313725,0.905882,0.298039,0.235294,0.92549,0.941176,0.945098,0.203922,0.596078,0.858824,0.160784,0.501961,0.72549&swatchOrder=0,1,2,3,4
import scipy
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import psycopg2
from collections import defaultdict
from collections import Counter
import xgboost as xgb

import json
import plotly

import datetime

con = psycopg2.connect(database='codeforces', user='Joy')
cur = con.cursor()

solarized = dict(
        darkblue  ='rgba(0, 43, 54, .7)',
        darkgray  ='rgba(88, 110, 117, .7)',
        gray ='rgba(101, 123, 131, .7)',
        lightgray ='rgba(131, 148, 150, .7)',
        beige ='rgba(238, 232, 213, .7)',
        lightbeige ='rgba(253, 246, 227, .7)',
        yellow  ='rgba(181, 137, 0, .7)',
        orange ='rgba(203, 75, 22, .7)',
        red ='rgba(220, 50, 47, .7)',
        magenta ='rgba(211, 54, 130, .7)',
        violet ='rgba(108, 113, 196, .7)',
        blue ='rgba(38, 139, 210, .7)',
        cyan ='rgba(42, 161, 152, .7)',
        green ='rgba(133, 153, 0, .7)',
        )
def smooth_ratings(df):

    window = df.shape[0]/3
    window += ((window%2) == 0)

    smooth_rating = scipy.signal.savgol_filter(df.newrating.values, window, polyorder=1, axis=0)

    return smooth_rating

def get_shapes(x0, x1):
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

    return shapes

def get_prate_hist(problems):
    prate_contest =  problems.loc[problems.participanttype == 'CONTESTANT', 'problem_rating']
    prate_practice = problems.loc[problems.participanttype != 'CONTESTANT', 'problem_rating']
    _, bins = np.histogram(prate_contest, bins=30)
    trace_histogram_contest = dict(
        x=prate_contest,
        name="contest",
	titlefont=dict(
	  size= 12
	),
        opacity=0.5,
        type='histogram',
        autobinx=False,
        xbins = dict(
          end=max(bins), 
          size= (max(bins) - min(bins))/20.0, 
          start=min(bins)
        ),
        marker=dict( color= solarized['violet'], size=10,),
    )
    trace_histogram_practice = dict(
        x=prate_practice,
        name="practice",
	titlefont=dict(
	  size= 12
	),
        opacity=0.5,
        type='histogram',
        autobinx=False,
        xbins = dict(
          end=max(bins), 
          size= (max(bins) - min(bins))/20.0, 
          start=min(bins)
        ),
        marker=dict( color= solarized['yellow'], size=10,),
    )

    hist_data = [
        trace_histogram_contest,
        trace_histogram_practice
        ]

    hist_plot = dict(
        data=hist_data,
        layout=dict(
            barmode="overlay",
            showlegend=False,
            legend=dict(orientation="h"),
            xaxis=dict(
                title="Problem difficulty",
		titlefont=dict(
		  size= 12,
		),
            ),
            yaxis=dict(
                title="Number of problems",
		titlefont=dict(
		  size= 12,
		),
            ),
            autosize = "True",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict( l=50, r=10, b=30, t=0, pad=0),
            hovermode='closest',
        )
    )
    return hist_plot

def get_user_problem_hist(problems_solved, handle, highgrowth_users):
    problems_solved.set_index('handle', inplace=True)
    counts = problems_solved['count']
    hist, bins = np.histogram(counts, bins=20)
    trace_allusers = dict(
        x=counts,
        type='histogram',
        histnorm='probability',
        marker=dict( color= solarized['magenta'], size=10,),
        autobinx=False,
        xbins = dict(
          end=max(bins), 
          size= (max(bins) - min(bins))/20.0, 
          start=min(bins)
        ),
        name="all users within 100 points"
    )

    trace_highgrowth = dict(
        x=problems_solved.loc[highgrowth_users, 'count'],
        type='histogram',
        histnorm='probability',
        marker=dict( color= solarized['blue'], size=10,),
        autobinx=False,
        xbins = dict(
          end=max(bins), 
          size= (max(bins) - min(bins))/20.0, 
          start=min(bins)
        ),
        name="users who improved"
    )

    usr_cnt = problems_solved.loc[handle, 'count']
    trace_user = dict(
        x=[usr_cnt],
        y=[0],
        type='scatter',
        marker=dict( color= solarized['darkblue'], size=10,),
        mode='markers',
        name="you"
    )

    user_problems_solved = dict(
        data=[trace_allusers, trace_highgrowth, trace_user],
        layout=dict(
            barmode="overlay",
            showlegend=True,
            autosize = "False",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict( l=60, r=0, b=30, t=0, pad=0),
            xaxis=dict(title="Number of problems",
                    showgrid=False,
		    titlefont=dict(
		    size= 12,
		    color= '#7f7f7f'
		    ),
	    ),
            yaxis=dict(title="Number of users",
              showgrid=False,
		titlefont=dict(
		  size= 12,
		  color= '#7f7f7f'
		),
	    )
        )
    )
    return user_problems_solved

def get_user_contest_hist(ncontests, handle, highgrowth_users):
    ncontests.set_index('handle', inplace=True)
    counts = ncontests['count']
    hist, bins = np.histogram(counts, bins=20)

    trace_ncontests = dict(
        x=counts,
        type='histogram',
        marker=dict( color= solarized['magenta'], size=10,),
        histnorm='probability',
        autobinx=False,
        xbins = dict(
          end=max(bins), 
          size= (max(bins) - min(bins))/20.0, 
          start=min(bins)
        ),
        name="all users within 100 points"
    )

    trace_highgrowth = dict(
        x=ncontests.loc[highgrowth_users, 'count'],
        type='histogram',
        marker=dict( color= solarized['blue'], size=10,),
        histnorm='probability',
        autobinx=False,
        xbins = dict(
          end=max(bins), 
          size= (max(bins) - min(bins))/20.0, 
          start=min(bins)
        ),
        name="users who improved"
    )

    usr_cnt = ncontests.loc[handle, 'count']
    usr_ncontests = dict(
        x=[usr_cnt],
        y=[0],
        type='scatter',
        marker=dict( color= solarized['darkblue'], size=10,),
        mode='markers',
        name="you"
    )

    user_contest_counts = dict(
        data=[trace_ncontests, trace_highgrowth, usr_ncontests],
        layout=dict(
            barmode="overlay",
            showlegend=True,
            autosize = "False",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict( l=60, r=0, b=30, t=0, pad=0),

            xaxis=dict(title="Number of contests",
                showgrid=False,
                titlefont=dict( size= 12, color= '#7f7f7f'),
	    ),

            yaxis=dict(
                title="Number of users",
                showgrid=False,
		titlefont=dict( size= 12, color= '#7f7f7f'),
	    ),
        )
    )
    return user_contest_counts

def get_tags(tags):
    counts = Counter(list(tags)).most_common()
    x = [a[0] for a in counts]
    y = [a[1] for a in counts]
    trace = dict(
        x=y,
        y=x,
        type='scatter',
        mode='markers',
        marker=dict(
            color=solarized['blue'],
            size=np.clip(np.array(y)/2.0, 5, 15),
	)
    )

    user_tags = dict(
        data=[trace],
        showLink=False,
        editable=False,
        layout=dict(
            showlegend=False,
            autosize = "False",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict( l=150, r=10, b=50, t=10, pad=0),
            yaxis=dict(
                showgrid=True,
                dtick=1,
                tickfont=dict(
                        size=12,
                    )
                ),
            xaxis=dict(title="Number of problems solved",
                showgrid=False,
		titlefont=dict(
		  size= 12,
		  color= '#7f7f7f'
		),
	    ),
	)
    )

    return user_tags

def plot_user_rating(user_rating, problems, problems_solved, handle, ncontests, tags, highgrowth_users):
    now = pd.to_datetime(datetime.date.today())

    try:
        smoothed = smooth_ratings(user_rating)
    except:
        smoothed = None

    # predictions
    x_pred =[now]
    last_rating = user_rating.newrating.values[-1]

    # for drawing squares
    x0 = min(problems.starttimeseconds)
    x1 = now
    shapes = get_shapes(x0, now)


    # ------------------------------------------------------
    # problem rating
    problems['problem_rating'] = np.clip(problems['problem_rating'], 0, 4000)
    problem_annotation = problems.contestname.str.cat(problems.problemid, sep=' ')
    psizes = 1.0/problems.solve_probability
    psizes = np.clip(psizes, 5, 20)
    trace_problemssolved = dict(
        x=problems.starttimeseconds,
        y=problems.problem_rating,
        mode='markers',
        text=problem_annotation,
        marker=dict(
            color='rgba(231, 76, 60, .5)',
            size=psizes,
            ),
        type='scatter',
        name="problems solved"
    )

    # ------------------------------------------------------
    # user rating
    trace_userrating = dict(
        x=user_rating.ratingupdatetimeseconds,
        y=user_rating.newrating,
        mode='markers',
        text=user_rating.contestname,
        marker=dict(
            color=solarized['blue'],
            size=10,
            ),
        type='scatter',
        name="contest performance"
    )

    # ------------------------------------------------------
    # smoothed user rating
    trace_smooth  = None
    if smoothed is not None:
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

    # ------------------------------------------------------
    # histograms
    hist_plot = get_prate_hist(problems)
    user_problems_solved = get_user_problem_hist(problems_solved, handle, highgrowth_users)
    user_contest_counts = get_user_contest_hist(ncontests, handle, highgrowth_users)

    # ------------------------------------------------------
    # tags
    user_tags = get_tags(tags)

    # ------------------------------------------------------
    # all rating data
    rating_data = [a for a in [
        trace_problemssolved,
        trace_userrating,
        trace_smooth] if a is not None]
    rating_plot = dict(
        data=rating_data,
        layout=dict(
            showlegend=True,
            legend=dict(orientation="h"),
            xaxis=dict(
                range=[x0, x1],
                domain=[0, 1],
		titlefont=dict(
		  size= 12,
		  color= '#7f7f7f'
		),
            ),
            yaxis=dict(
                range=[min(1000, min(user_rating.newrating)-100), max(2000, max(problems.problem_rating[problems.problem_rating < 5000])+200)],
                title="User/ problem rating",
		titlefont=dict(
		  size= 12,
		  color= '#7f7f7f'
		),
            ),
            subplot_titles=('1', '2', '3', '4'),
            autosize = "False",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(
                    l=60,
                    r=10,
                    b=0,
                    t=10,
                    pad=0
                ),
            shapes=shapes,
            hovermode='closest',
        )
    )

    # -----------------------------------------------------------------------------
    graphs = [p for p in [rating_plot, user_problems_solved, user_contest_counts, user_tags, hist_plot ] if p is not None]

    # Add "ids" to each of the graphs to pass up to the client for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON

def predict(model, xgb_x_df):
    missing_val = -99999
    xgb_x_df.drop(['handle', 'contestid'], axis=1, inplace=True)
    xgb_x = xgb.DMatrix(xgb_x_df, missing = missing_val)
    orig_delta = model.predict(xgb_x)[0]

    
    print "=================================================================="
    print orig_delta
    print "=================================================================="

    q = """ 
    select tag from all_tags
    """
    all_tags = pd.read_sql(q, con)['tag']
    all_tags = set(all_tags)
    
    all_tags.discard("2-sat")
    all_tags.discard("fft")
    all_tags.discard("*special")
    all_tags.discard("chinese remainder theorem")

    # -------------------------------------------------------
    # Figure out optimal difficulty
    # -------------------------------------------------------
    difficulties = ['n100', 'n200', 'n300', 'n400', 'n500']
    e = xgb_x_df.copy()
    e['n_solved'] += 10
    e['total_problems'] += 10

    best_val = -float("Inf")
    best_diff = ''
    for t in difficulties:
	e[t] += 10
	delta = model.predict(xgb.DMatrix(e))[0] - orig_delta
	if delta > best_val:
	    best_val = delta
	    best_diff = t
    best_diff = best_diff.replace("n", "")

    best_diff_str = "around " + str(best_diff) + " points higher than your rating."

    # -------------------------------------------------------
    # Figure out optimal tags
    # -------------------------------------------------------
    good_tags = []
    bad_tags = []
    for t in list(all_tags):
	e = xgb_x_df.copy()
	e[t] += 100
	delta = model.predict(xgb.DMatrix(e))[0] - orig_delta
	if  delta > 0:
	    good_tags.append((t, delta))
	elif delta < 0:
	    bad_tags.append((t, delta))

    good_tags.sort(key=lambda x: x[1], reverse=True)
    bad_tags.sort(key=lambda x: x[1], reverse=True)

    max_display = 8
    if len(good_tags) > max_display:
        good_tags = good_tags[0:max_display]
    if len(bad_tags) > max_display:
        bad_tags = bad_tags[0:max_display]


    if orig_delta > 0:
        delta_str = '+'
    if orig_delta < 0:
        delta_str = ''
    delta_str += str(orig_delta)


    return int(best_diff), best_diff_str, delta_str, good_tags, bad_tags

def filter_problems(handle, problem_thresh, problem_inc, highgrowth_users):
    query = """
	SELECT DISTINCT ON(problem_info.name)
            problem_rating.problemrating,
            problem_info.contestid,
            problem_info.problemid,
            problem_info.contestname,
            problem_info.name,
            problem_info.division,
            problem_info.points,
            tags.tag
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
            INNER JOIN tags
                ON
                    problem_info.contestid = tags.contestid
                    AND
                    problem_info.problemid = tags.problemid
        ORDER BY problem_info.name, problem_rating.problemrating DESC
        """ % (problem_thresh, problem_thresh + problem_inc)
    problem_suggest = pd.read_sql_query(query,con)

    if len(problem_suggest) == 0:
	query = """
	    SELECT DISTINCT ON(problem_info.name)
		problem_rating.problemrating,
		problem_info.contestid,
		problem_info.problemid,
		problem_info.contestname,
		problem_info.name,
		problem_info.division,
		problem_info.points,
		tags.tag
	    FROM problem_info
		INNER JOIN problem_rating
		    ON 
			problem_info.contestid = problem_rating.contestid
			AND
			problem_info.problemid = problem_rating.problemid
			AND
			problem_info.division = '1'
			AND
			problem_rating.problemrating >= %d
		INNER JOIN tags
		    ON
			problem_info.contestid = tags.contestid
			AND
			problem_info.problemid = tags.problemid
	    ORDER BY problem_info.name, problem_rating.problemrating DESC
	    """ % (problem_thresh)
	problem_suggest = pd.read_sql_query(query,con)

    # problems solved
    query = """
    SELECT contestid, problemid FROM probability_solve
	WHERE
	    handle='%s'
    """ % handle 
    cur.execute(query)
    done_ids = cur.fetchall()
    cids = set([c[0] for c in done_ids])
    pids = set([c[1] for c in done_ids])

    filt = np.logical_or(~problem_suggest.contestid.isin(cids), ~problem_suggest.problemid.isin(pids))

    problem_suggest['contestname'] = problem_suggest['contestname'].apply(lambda x: x.decode('utf-8','ignore'))
    problem_suggest['name'] = problem_suggest['name'].apply(lambda x: x.decode('utf-8','ignore'))

    problem_suggest = problem_suggest.loc[filt]
    return problem_suggest.to_dict(orient='records')







