This folder contains code for codeforces API calls, data parsing and cleaning, as well as scripts for scraping problem statements from codeforces and topcoder (not used for final model).

## Requirements
- Python 2.7
- PostgreSQL 9.6.3
- Python packages:
	- numpy
	- pandas
	- XGBoost
	- Flask
	- psycopg2
	

## Files



### Problem statements
Scripts for scraping problem statements from topcoder and codeforces (not provided by API):


- ```api_get_problem_statements-topcoder.ipynb```

- ```api_get_problem_statements_codeforces.ipynb```


### API calls, inferring problem difficulty


- ```update_data.sh```: bash script wrapper for updater.py

- ```updater.py```: wrapper that connects functions in elo.py and api_functions. Converts resulting pandas dataframes from API calls to codeforces (returned by api_functions) into problem difficulty ratings (elo.py).

- ```api_functions.py```: calls codeforces API and returns pandas dataframes

- ```elo.py```: runs a binary search algorithm to infer problem rating (difficulty) from contest and problem information


### User submissions and other information


- ```api_get_allUserInformation.ipynb```: functions for retrieving all user submission information and rating histories. Since these scripts typically take hours to days to finish, this code provides ways to re-start the script at the previous save point in case of uncaught errors.

- ```api_allUserRatingHistories_RNN.py```: a modified version of ```api_get_allUserInformation.ipynb``` for use in recurrent neural networks.

- ```api_get_contest_hacks.ipynb```: gets contest hack information from codeforces. This information was not used in the final model.


### Inserting into SQL


- ```api_to_sql.ipynb```: inserting pandas dataframes into SQL databases

- ```sql_fixes.ipynb```: setting indices and fixing data types on SQL database to streamline SQL calls further down the line.


### Others

- ```calc_probabilities.py```: For feature engineering: figures out the probability of a person solving a problem based on ELO scores.

- ```train_val_test_split.ipynb```: stratified splitting for training, validation, and test sets.
