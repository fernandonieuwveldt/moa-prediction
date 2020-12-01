
"""
workflow for training and predicting
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas
import numpy
from moa.classifier import MOAClassifier


train_data_features = pandas.read_csv("moa/data/lish-moa/train_features.csv").drop('sig_id', axis=1)
test_data_features = pandas.read_csv("moa/data/lish-moa/test_features.csv").drop('sig_id', axis=1)
train_data_features['cp_time'] = train_data_features['cp_time'].map(str)
test_data_features['cp_time'] = test_data_features['cp_time'].map(str)
raw_labels = pandas.read_csv("moa/data/lish-moa/train_targets_scored.csv").drop('sig_id', axis=1)


ensemble_prediction = []
number_ensembles = 3
for _ in range(number_ensembles):
    model = MOAClassifier(batch_size=32, epochs=100)
    model.fit(train_data_features, raw_labels)
    ensemble_prediction.append(model.predict(test_data_features))


def submit_run(file_name=None):
    submit_data_frame = pandas.read_csv("moa/data/lish-moa/sample_submission.csv")
    submit_data_frame.iloc[:, 1:] = numpy.mean(ensemble_prediction, axis=0)
    submit_data_frame.to_csv(f'moa/data/lish-moa/{file_name}.csv', index=False)


submit_run('test_results')
