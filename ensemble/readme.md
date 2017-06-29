This folder contains code for fitting various ensemble regressors.

## Requirements
- Python 2.7
- PostgreSQL 9.6.3
- Python packages:
	- numpy
	- pandas
	- psycopg2
	- matplotlib
	- seaborn
	- xgboost
	- sklearn
	

## Files

### Features


- ```generate_features_ensemble.py```: extensive feature engineering for generating ensemble features. Uses a queue of submission events to look backwards in time anytime a contest is added to the queue. Parallelized for an 8-core Google Compute Engine instance. Takes around 3 hours to complete.

- ```ensemble_functions.py```: utility script for parsing features to feed into ensemble fitting models


### Model fitting

- ```random_forest_regressor.ipynb```: jupyter notebook for preliminary fitting to a RF regressor

- ```rf_regressor_gridsearch.py```: parallel gridsearch using sklearn's ```gridsearchCV``` function

- ```gradient_boosted_regression.ipynb```: Fitting a gradient-boosted regressor using XGBoost.




### For fun: predicting user rating based on solved problems
Fitting a random forest regressor on user rating based on a binary matrix of problems solved by each user. This was done just for fun, since such a model would not be great for production (we would need to re-fit it every time a new problem is released!), and doesn't really give great insight. The model achives surprisingly good fit (r^2 between 0.6 and 0.8), It appears to infer problem difficulty. Includes some tSNE visualizations.

- ```RF_on_questions.ipynb```: uses all problems a user solved. r^2 = 0.8. I believe this fits well because it's inferring problem difficulty during contests.

- ```RF_on_questions_practiceonly.ipynb```: uses only practice problems, still achives a decent r^2 value of 0.63.

