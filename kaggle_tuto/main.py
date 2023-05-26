#!/usr/bin/env python3

import pandas as pd
import numpy as np


features = [
    'radius',
    'texture',
    'perimeter',
    'area',
    'smoothness',
    'compactness',
    'concavity',
    'concave points',
    'symmetry',
    'fractal dimension'
]

columns_names = []
columns_names.append('id')
columns_names.append('diagnosis')
for feature in features:
    columns_names.append(feature + '_mean')
    columns_names.append(feature + '_se')
    columns_names.append(feature + '_worst')


# Load the dataset
dataset = pd.read_csv('../data.csv', header=None)

# Name the columns
dataset.columns = columns_names

# Drop the 'id' column (1st column)
dataset = dataset.drop('id', axis=1)

################################################################################

benign_data = dataset[dataset['diagnosis'] == 'B']
malignant_data = dataset[dataset['diagnosis'] == 'M']

# Separate the datasets into training and testing sets
benign_data.shuffle()
malignant_data.shuffle()

# 80% of the data for training
benign_training_data = benign_data[:int(len(benign_data) * 0.8)]
malignant_training_data = malignant_data[:int(len(malignant_data) * 0.8)]

# 20% of the data for testing
benign_testing_data = benign_data[int(len(benign_data) * 0.8):]
malignant_testing_data = malignant_data[int(len(malignant_data) * 0.8):]

# Concatenate the training and testing data
training_data = pd.concat([benign_training_data, malignant_training_data])
testing_data = pd.concat([benign_testing_data, malignant_testing_data])

# Shuffle the data
training_data.shuffle()
testing_data.shuffle()

# Separate the features from the labels
training_features = training_data.drop('diagnosis', axis=1)
training_labels = training_data['diagnosis']

testing_features = testing_data.drop('diagnosis', axis=1)
testing_labels = testing_data['diagnosis']

# Convert the labels to 0 and 1
training_labels = training_labels.replace('B', 0)
training_labels = training_labels.replace('M', 1)

testing_labels = testing_labels.replace('B', 0)
testing_labels = testing_labels.replace('M', 1)

# Convert the features and labels to numpy arrays
training_features = training_features.to_numpy()
training_labels = training_labels.to_numpy()

testing_features = testing_features.to_numpy()
testing_labels = testing_labels.to_numpy()

################################################################################


