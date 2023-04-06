#!/usr/bin/env python3

import tools
import MyMLP as mlp
import numpy as np
import pandas as pd

# prepare data

try:
    data = pd.read_csv('data.csv')
except:
    print('data.csv not found')
    exit()

# extract diagnosis/id column
ids = np.array(data.iloc[:, 0]).reshape(-1, 1)
diagnosis = np.array(data.iloc[:, 1]).reshape(-1, 1)
# delete diagnosis/id column
data = data.drop(data.columns[[0, 1]], axis=1)
data = np.array(data, dtype=float)
# transform diagnosis column into a vector of 0 and 1
diagnosis = np.where(diagnosis == 'M', 1, 0)
# split data into train and test
x_train, x_test, y_train, y_test = tools.data_splitter(data, diagnosis, 0.6)

# create network
nb_inputs = data.shape[1]
hidden_layer_sizes = [3, 4]
mymlp = mlp.MyMLP(hidden_layer_sizes,
                  [np.random.rand(3, nb_inputs), np.random.rand(
                      4, 3), np.random.rand(2, 4)],
                  [np.random.rand(3, 1), np.random.rand(
                      4, 1), np.random.rand(2, 1)],
                  'sigmoid',
                  0.1,
                  1000)

predictions = mymlp.predict(x_train[0].reshape(-1, 1))
print(predictions)
