from typing import List
import numpy as np
import pandas as pd
import tools


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


class Network:
    def __init__(self, layers):
        self.layers = layers


'''
Each layer is composed of neurons.
Each layer has a weight matrix and a bias matrix.
The weight matrix is a matrix of size (number of neurons in the layer, number of neurons in the previous layer).
The bias matrix is a matrix of size (number of neurons in the layer, 1).
'''


class Layer:
    def __init__(self, neurons, weights: np.ndarray, biases: np.ndarray):
        self.neurons = neurons
        self.weights = weights
        self.biases = biases


'''
Multi-Layer Perceptron

Version used exclusively to predict boolean values.

The MLP is composed of a network of layers, each layer is composed of neurons.
The MLP is trained using Backpropagation.
The MLP is trained using Stochastic Gradient Descent.
The input layer should be given as a parameter in a prediction or score function.


'''


class MyMLP:
    def __init__(self, hidden_layer_sizes, weight_matrices, bias_matrices,
                 activation, learning_rate_init, max_iter):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.weight_matrices = weight_matrices
        self.bias_matrices = bias_matrices
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter

        if self.activation == 'sigmoid':
            self.activation_function = sigmoid
        elif self.activation == 'relu':
            self.activation_function = ReLU
        else:
            print('in "init" function: activation function is not valid')
            raise ValueError

        if self.learning_rate_init <= 0:
            print('in "init" function: learning rate is not valid')
            raise ValueError

        if self.max_iter <= 0:
            print('in "init" function: max iteration is not valid')
            raise ValueError

        if len(weight_matrices) != len(bias_matrices):
            print(
                'in "init" function: weight matrices and bias matrices do not share compatible dimensions')
            raise ValueError
        if len(weight_matrices) != len(hidden_layer_sizes) + 1:
            print(
                'in "init" function: weight matrices and hidden layer sizes do not share compatible dimensions')
            raise ValueError

        self.network = self.create_network()

    def create_layer(self, size, values, weights: np.ndarray,
                     biases: np.ndarray):
        neurons = []
        if size != weights.shape[0]:
            print(
                'in "create_layer" function: weights matrix and layer size do not share compatible dimensions')
            raise ValueError
        if weights.shape[0] != biases.shape[0]:
            print(
                'in "create_layer" function: weights matrix and biases matrix do not share compatible dimensions')
            raise ValueError
        for _ in range(size):
            neurons.append(np.random.uniform(-1, 1))
        return Layer(np.array(neurons).reshape(-1, 1), weights, biases)

    def create_network(self):
        # create network
        network = Network([])
        for i in range(len(self.hidden_layer_sizes)):
            layer_size = self.hidden_layer_sizes[i]
            layer = self.create_layer(self.hidden_layer_sizes[i],
                                      np.random.rand(
                                          1, self.hidden_layer_sizes[i]),
                                      self.weight_matrices[i],
                                      self.bias_matrices[i])
            network.layers.append(layer)

        # last layer: output layer
        layer_size = 2
        last_layer = len(self.hidden_layer_sizes)
        layer = self.create_layer(layer_size,
                                  [0, 0],
                                  self.weight_matrices[last_layer],
                                  self.bias_matrices[last_layer])
        network.layers.append(layer)
        return network

    def predict(self, x):
        '''
        Predicts the output of the MLP for the given input.
        x is a numpy array of size (number of features, 1).
        '''

        # check input
        if x.shape[0] != self.network.layers[0].weights.shape[1]:
            print(
                'in "predict" function: input and network do not share compatible dimensions')
            raise ValueError
        if x.shape[1] != 1:
            print(
                'in "predict" function: prediction is not valid for multiple inputs')
            raise ValueError

        # forward propagation
        for i in range(len(self.network.layers)):
            layer = self.network.layers[i]
            entry = None
            if i == 0:
                entry = x
            else:
                entry = self.network.layers[i - 1].neurons

            layer.neurons = self.activation_function(
                layer.weights @ entry + layer.biases)

        # return output
        return np.array(self.network.layers[-1].neurons).reshape(-1, 1)

    def loss(self, y, y_pred):
        '''
        Computes the loss of the MLP for the given input and output.
        y is a numpy array of size (number of outputs, 1).
        y_pred is a numpy array of size (number of outputs, 1).
        '''
        return np.sum((y - y_pred) ** 2)

    def score(self, x, y):
        '''
        Computes the score of the MLP for the given input and output.
        x is a numpy array of size (number of features, number of samples).
        y is a numpy array of size (number of outputs, number of samples).
        '''
        y_pred = []
        for i in range(x.shape[0]):
            diag = self.predict(x[i, :].reshape(-1, 1))
            y_pred.append(0 if diag[0] > diag[1] else 1)
        y_pred = np.array(y_pred).reshape(-1, 1)
        return 1 - self.loss(y, y_pred)

    def fit(self, x, y):
        '''
        Trains the MLP for the given input and output.
        x is a numpy array of size (number of samples, number of features).
        y is a numpy array of size (number of samples, number of outputs).
        '''
        for epoch in range(self.max_iter):
            print(f'epoch {epoch}/{self.max_iter} - score: {self.score(x, y)}')
            for i in range(x.shape[0]):
                self.train(x[i, :].reshape(-1, 1), y[i, :].reshape(-1, 1))

    def train(self, x, y):
        '''
        Trains the MLP for the given input and output.
        x is a numpy array of size (number of features, 1).
        y is a numpy array of size (number of outputs, 1).
        '''
        # forward propagation
        for i in range(len(self.network.layers)):
            layer = self.network.layers[i]
            entry = None
            if i == 0:
                entry = x
            else:
                entry = self.network.layers[i - 1].neurons

            layer.neurons = self.activation_function(
                layer.weights @ entry + layer.biases)

        # backward propagation
        for i in range(len(self.network.layers) - 1, -1, -1):
            layer = self.network.layers[i]
            entry = None
            if i == 0:
                entry = x
            else:
                entry = self.network.layers[i - 1].neurons

            if i == len(self.network.layers) - 1:
                # output layer
                layer.weights = layer.weights - self.learning_rate_init * (
                    2 * (layer.neurons - y) * layer.neurons *
                    (1 - layer.neurons) @ entry.T)
                layer.biases = layer.biases - self.learning_rate_init * (
                    2 * (layer.neurons - y) * layer.neurons *
                    (1 - layer.neurons))
            else:
                # hidden layers
                layer.weights = layer.weights - self.learning_rate_init * (
                    2 * (layer.neurons - y) * layer.neurons *
                    (1 - layer.neurons) @ entry.T)
                layer.biases = layer.biases - self.learning_rate_init * (
                    2 * (layer.neurons - y) * layer.neurons *
                    (1 - layer.neurons))
