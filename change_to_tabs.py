import pandas as pd
df = pd.read_csv('all_submissions.csv', engine='c')
df.to_csv('all_submissions.tsv', index=False, header=True, encoding='utf-8', sep='\t')
    