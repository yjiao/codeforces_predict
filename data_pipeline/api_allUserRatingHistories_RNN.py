# this is a one-off script for getting data for training recurrent neural
# networks, not the general data pipeline

import requests
import pandas as pd
from os import listdir
import time
import sys
from os.path import exists

print "reading handles..."
with open('all_handles.txt') as f:
    hdl = f.readlines()
print "done"

hdl = [s.strip() for s in hdl]

rating_changes = []

cnt = 0
maxtries = 5
wait = 10
filename = 'all_ratingHistories.csv'
start = 1252

print "processing %d handles, starting at %s" % (len(hdl), hdl[start])
for handle in hdl[start:]:
    tries = 0
    while (tries < maxtries):
        try:
            url = 'http://codeforces.com/api/user.rating?handle=' + handle

            r = requests.get(url).json()['result']
            rating_changes.extend(r)

            cnt += 1
            
            if (cnt%10==0):
                print '%d of %d users processed' % (cnt, len(hdl))
                df = pd.DataFrame.from_dict(rating_changes)
                df.drop(labels = ['contestName', 'rank', 'ratingUpdateTimeSeconds'], axis=1, inplace=True)
                df.to_csv(filename, index=False, mode='a', header=(not exists(filename)), encoding='utf-8')
                rating_changes = []
            break

        except:
            print "error encountered while processing url:", url

            if tries > maxtries:
                print "max tries exceeded, aborting"
                sys.exit()

            print "waiting %d sec before trying again" % (wait)
            time.sleep(wait)
            tries += 1

df = pd.DataFrame.from_dict(rating_changes)
df.drop(labels = ['contestName', 'rank', 'ratingUpdateTimeSeconds'], axis=1, inplace=True)
df.to_csv('all_ratingHistories.csv', index=False)
#   contestId                contestName   handle  newRating  oldRating  rank  \
#0          2   Codeforces Beta Round #2  tourist       1602       1500    14   
#1          8   Codeforces Beta Round #8  tourist       1764       1602     5   
#2         10  Codeforces Beta Round #10  tourist       1878       1764    18   
#3         13  Codeforces Beta Round #13  tourist       1967       1878    11   
#4         19  Codeforces Beta Round #19  tourist       2063       1967     2   
#
#   ratingUpdateTimeSeconds  
#0               1267124400  
#1               1270748700  
#2               1271353500  
#3               1273161600  
#4               1277398800  