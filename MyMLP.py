import numpy as np
import pandas as pd
import tools


class MyMLP:
    def __init__(self, hidden_layer_sizes, activation,
                 learning_rate_init, max_iter, tol, verbose):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, x_train, y_train):
        # check if x and y are valid
        if not tools.is_matrix_valid(x_train) or not tools.is_matrix_valid(y_train):
            print('in "fit" function: x or y is not valid')
            return None
        if x_train.shape[0] != y_train.shape[0]:
            print('in "fit" function: x and y do not share compatible dimensions')
            return None

        # create the network
        self.network = self.create_network(x_train, y_train)

        # train the network
        self.train_network(x_train, y_train)

    def predict(self, x_test):
        # check if x is valid
        if not tools.is_matrix_valid(x_test):
            print('in "predict" function: x is not valid')
            return None

        # predict
        return self.predict_network(x_test)
    
    def score(self, x_test, y_test):
        # check if x and y are valid
        if not tools.is_matrix_valid(x_test) or not tools.is_matrix_valid(y_test):
            print('in "score" function: x or y is not valid')
            return None
        if x_test.shape[0] != y_test.shape[0]:
            print('in "score" function: x and y do not share compatible dimensions')
            return None

        # predict
        y_pred = self.predict_network(x_test)

        # calculate score
        score = self.calculate_score(y_pred, y_test)
        return score
    
    def create_network(self, x_train, y_train):
        # create the network
        network = []
        # input layer
        network.append(self.create_layer(x_train.shape[1], self.hidden_layer_sizes[0]))
        # hidden layers
        for i in range(1, len(self.hidden_layer_sizes)):
            network.append(self.create_layer(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i]))
        # output layer
        network.append(self.create_layer(self.hidden_layer_sizes[-1], y_train.shape[1]))
        return network
    
    def create_layer(self, n_inputs, n_neurons):
        # create the layer
        layer = []
        # neurons
        for i in range(n_neurons):
            layer.append(self.create_neuron(n_inputs))
        return layer
    
    def create_neuron(self, n_inputs):
        # create the neuron
        neuron = []
        # weights
        for i in range(n_inputs):
            neuron.append(self.create_weight())
        # bias
        neuron.append(self.create_weight())
        return neuron
    
    def create_weight(self):
        # create the weight
        return np.random.normal()
    
