
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/01_home.png?raw=true =600x)

This repository hosts a collection of scripts used for generating the model behind [Code Coach](codecoach.fun), my project at Insight Data Science. This project uses an ensemble regressor trained on 20 million codeforces submissions to predict future rating changes based on past user behavior, and recommends problems on Codeforces for users to solve. Problem ratings (difficulties) were previously calculated [here](https://github.com/yjiao/codeforces-api). More information about the models can be found at [Code Coach](codecoach.fun).


While the code for training the ensemble regressors is desposited here, csv files and SQL databases were not uploaded due to size constraints. Additionally, while the data used for this project is publically available through the codeforces API, I wish to respect the privacy of codeforces users and not upload everyone's submission histories onto github.



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


## Usage
Sample output from the codecoach.fun website is provided below.

![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/02a_search.png?raw=true =600x)

The landing page for a random user (handle erased) is shown here, with a next predicted rating change.
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/02b_ratingchange.png?raw=true =600x)

Codecoach provides dashboards for the user to monitor practice progress.
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/03_rating_history.png?raw=true =600x)
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/04_practice_v_contest.png?raw=true =600x)
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/05_tags.png?raw=true =600x)
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/06_compare.png?raw=true =600x)

Codecoach provides practice suggestions.
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/07_suggest.png?raw=true =600x)
![](https://github.com/yjiao/codeforces_predict/blob/master/screenshots/08_suggest.png?raw=true =600x)


## History
2017 06 25 Initial public commit