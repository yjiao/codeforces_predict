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
	- sklearn
	- statsmodels
	

## Files

### Features
- ```smooth_features_no_lookahead.ipynb```: time-aware data smoothing, looking only at past data to avoid lookahead bias

- ```feature_functions.py```: utilities for calculating features

- ```generate_features_OLS.py```: generates features for linear models

- ```generate_last_training_examples.py```: get a list of last training example for each user for use in a toy example showcasing how the model works (not used in production)


### Model fitting

- ```initial_OLS.ipynb```: first attempt at fitting model using OLS

- ```piecewise_OLS_timeaware.ipynb```: initial fitting a piece-wise linear model on smoothed user rating changes, based on current user rating

- ```piecewise_linear_regression_historical_and_recent_features.ipynb```: final piece-wise linear regression, uses more features than the previous version and achieves a better fit on the test data

- ```linear_regression_ui_visualizations.ipynb```: functions for visualizing the linear regression model

