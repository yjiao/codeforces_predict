#!/bin/bash
# This script runs some python functions for updating user rating information
# and calculating problem ratings.

# Note not all the command line options for updater are being used right now, for first pass there's no reason to have non-standard file names

update=true

if $update
then
    echo "0. Updating data required for creating plots..."
    python updater.py
fi

echo "1. Checking that all required csv files exist..."

arr=("problem_data.csv" "problem_ratings.csv")
for fileName in "${arr[@]}"
do
    if [ ! -e $fileName ]
    then
	echo "   $fileName not found. Please call this function without the -n flag to update required csv files"
	exit
    else
	echo "   $fileName found"
    fi

done
