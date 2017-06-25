# Code Coach: ML-powered programming practice for Codeforces

This repository hosts a collection of scripts used for generating the model behind [Code Coach](codecoach.fun), my project at Insight Data Science. This project uses an ensemble regressor trained on 20 million codeforces submissions to predict future rating changes based on past user behavior, and recommends problems on Codeforces for users to solve. Problem ratings (difficulties) were previously calculated [here](https://github.com/yjiao/codeforces-api).

While the code for training the ensemble regressors is desposited here, csv files and SQL databases were not uploaded due to size constraints. Additionally, while the data used for this project is publically available through the codeforces API, I wish to respect the privacy of codeforces users and not upload everyone's submission histories onto github.


![](https://github.com/yjiao/codeforces-api/blob/master/ui/img_histogram.png?raw=true =600x)


## Requirements
Note that packages marked as "not in production" means that while these models were used during the project, they are not present in the final codecoach.fun website. for variosu reasons.

- Python 2.7
- PostgreSQL 9.6.3
- Python packages:
	- numpy
	- scipy
	- pandas
	- sklearn
	- XGBoost
	- Flask
	- Jupyter (for running notebooks)
	- (not in production) Keras
	- (not in production) Tensorflow
	- (not in production) Statsmodels



## History
2017 06 25 Initial public commit