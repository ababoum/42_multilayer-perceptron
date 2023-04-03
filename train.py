import tools
import numpy as np
import pandas as pd

# prepare data

try:
    data = pd.read_csv('data.csv')
except:
    print('data.csv not found')
    exit()

# extract diagnosis column (2nd column)
diagnosis = data.iloc[:, 1]
diagnosis = np.array(diagnosis).reshape(-1, 1)
# delete diagnosis column (2nd column)
data = data.drop(data.columns[1], axis=1)
data = np.array(data)
# split data into train and test
x_train, x_test, y_train, y_test = tools.data_splitter(data, diagnosis, 0.6)

