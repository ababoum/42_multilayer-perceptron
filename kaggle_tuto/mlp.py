import random
import numpy as np


class MultiLayerPerceptron():
    def __init__(self, network, input_layer_size=30, hidden_layer_size=15, output_layer_size=2, learning_rate=0.005, max_epochs=600, bias_hidden_value=0.0, bias_output_value=0.0, activation='ReLU'):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.bias_hidden_value = float(bias_hidden_value)
        self.bias_output_value = float(bias_output_value)
        self.activation = self.action_functions[activation]
        self.deriv = self.derivative_functions[activation]

        # Connecting the network with the model
        self.network = network
        self.input_layer = self.network[0]
        self.hidden_layer = self.network[1]
        self.output_layer = self.network[2]

        # Starting Bias and Weights
        self.weight_matrix_hidden = self.starting_weights(
            self.hidden_layer_size, self.input_layer_size)
        self.weight_matrix_output = self.starting_weights(
            self.output_layer_size, self.hidden_layer_size)
        self.bias_matrix_hidden = np.array(
            [self.bias_hidden_value for i in range(self.hidden_layer_size)]).reshape(-1, 1)
        self.bias_matrix_output = np.array(
            [self.bias_output_value for i in range(self.output_layer_size)]).reshape(-1, 1)
        self.classes_number = 2

    action_functions = {
        'sigmoid': (lambda x: 1 / (1 + np.exp(-x))),
        'tanh': (lambda x: np.tanh(x)),
        'ReLU': (lambda x: x * (x > 0)),
    }
    derivative_functions = {
        'sigmoid': (lambda x: x * (1 - x)),
        'tanh': (lambda x: 1 - x ** 2),
        'ReLU': (lambda x: 1 * (x > 0))
    }

    def starting_weights(self, x, y):
        return np.array([[random.random() for i in range(y)] for j in range(x)])

    def feed_forward(self, inputs) -> None:
        '''Updates the values of the neurons inside the layers of the network'''

        self.input_layer = inputs.reshape(-1, 1)
        self.hidden_layer = self.activation(
            np.matmul(self.weight_matrix_hidden, self.input_layer) + self.bias_matrix_hidden)
        self.output_layer = self.activation(
            np.matmul(self.weight_matrix_output, self.hidden_layer) + self.bias_matrix_output)

    def back_propagation(self, outputs) -> None:
        '''Updates the values of the weights and biases of each layer of the network'''

        # Calculate the error
        error = outputs.reshape(-1, 1) - self.output_layer

        # Calculate the delta
        delta_output = error * self.deriv(self.output_layer)

        # Calculate the error of the hidden layer
        error_hidden = np.matmul(self.weight_matrix_output.T, delta_output)

        # Calculate the delta of the hidden layer
        delta_hidden = error_hidden * self.deriv(self.hidden_layer)

        # Update the weights
        self.weight_matrix_output += self.learning_rate * \
            np.matmul(delta_output, self.hidden_layer.T)
        self.weight_matrix_hidden += self.learning_rate * \
            np.matmul(delta_hidden, self.input_layer.T)

        # Update the biases
        self.bias_matrix_output += self.learning_rate * delta_output
        self.bias_matrix_hidden += self.learning_rate * delta_hidden

    def fit(self, inputs, outputs):
        for epoch in range(self.max_epochs):
            for i in range(len(inputs)):
                self.feed_forward(inputs[i])
                self.back_propagation(outputs[i])
            print(f'Epoch: {epoch} - Score {self.score(inputs, outputs)}')

    def predict(self, inputs):
        self.feed_forward(inputs)
        return self.output_layer

    def predict_classes(self, inputs):
        self.feed_forward(inputs)
        return np.argmax(self.output_layer)

    def score(self, inputs, outputs):
        predictions = []
        testing_sample_size = len(inputs)
        for i in range(testing_sample_size):
            predictions.append(self.predict_classes(inputs[i]))
        return np.sum(predictions == outputs) / len(outputs)
