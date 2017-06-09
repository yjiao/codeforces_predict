import scipy
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import psycopg2
import tensorflow as ts
from collections import defaultdict

con = psycopg2.connect(database='codeforces', user='Joy')
cur = con.cursor()

def smooth_ratings(df):

    window = df.shape[0]/3
    window += ((window%2) == 0)

    print df.info()

    smooth_rating = scipy.signal.savgol_filter(df.newrating.values, window, polyorder=1, axis=0)

    return smooth_rating
