#!/bin/bash

name="codeforces_rnn_training"
BUCKET_NAME=$name
for file in $(find rnn_train -mmin -60); do
	if ! gsutil -q stat gs://codeforces_rnn_training/tourist.csv; then
		echo $file
		gsutil -m cp $file gs://$BUCKET_NAME/
	fi
done