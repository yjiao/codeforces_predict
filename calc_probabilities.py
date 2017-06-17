import pandas as pd
import numpy as np
from glob import glob
from sqlalchemy import *

def prob_solve(difference):
    return 1.0 / ( 1 + 10 ** ((difference) / 400.0) )

def get_prob(f, engine):
    df = pd.read_csv(f)
    if len(df) > 0:
        df['rating_diff'] = df.problem_rating - df.smoothed_3months
        df['solve_probability'] = df.rating_diff.apply(prob_solve)
        dfout = df[ [
                    'handle',
                    'contestid',
                    'problemid',
                    'solve_probability',
                    'smoothed_3months',
                    'problem_rating'
                ] ]
        dfout.drop_duplicates(inplace = True)

        insert_into_sql(dfout, engine)

def insert_into_sql(df, engine):
    df.to_sql('probability_solve', engine, if_exists='append', index=False)

if __name__ == "__main__":

    engine = create_engine('postgres://%s@localhost/%s'%("Joy","codeforces"))
    metadata = MetaData()
    probs = Table('probability_solve', metadata,
	Column('handle', String, primary_key=True),
	Column('contestid', String, primary_key=True),
	Column('problemid', String, primary_key=True),
	Column('solve_probability', Float),
	Column('smoothed_3months', Float, primary_key=True),
	Column('problem_rating', Float, primary_key=True)
    )
    probs.drop(engine, checkfirst=True)
    probs.create(engine)

    files = glob('rnn_train/*.csv')
    for f in files:
        print f
        get_prob(f, engine)
