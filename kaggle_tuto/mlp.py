import random
import numpy as np


class MultiLayerPerceptron():
    def __init__(self, params=None):
        if (params == None):
            self.input_layer = 30                      # Input Layer
            self.hidden_layer = 15                     # Hidden Layer
            self.output_layer = 2                      # Output Layer
            self.learning_rate = 0.005                 # Learning rate
            self.max_epochs = 600                      # Epochs
            self.bias_hidden_value = -1                # Bias HiddenLayer
            self.biasOutputValue = -1                  # Bias OutputLayer
            self.activation = self.action_functions['sigmoid']
            self.deriv = self.derivative_functions['sigmoid']
        else:
            self.inputLayer = params['InputLayer']
            self.hiddenLayer = params['HiddenLayer']
            self.OutputLayer = params['OutputLayer']
            self.learningRate = params['LearningRate']
            self.max_epochs = params['Epocs']
            self.BiasHiddenValue = params['BiasHiddenValue']
            self.BiasOutputValue = params['BiasOutputValue']
            self.activation = self.action_functions[params['ActivationFunction']]
            self.deriv = self.derivative_functions[params['ActivationFunction']]

        'Starting Bias and Weights'
        self.WEIGHT_hidden = self.starting_weights(
            self.hiddenLayer, self.inputLayer)
        self.WEIGHT_output = self.starting_weights(
            self.OutputLayer, self.hiddenLayer)
        self.BIAS_hidden = np.array(
            [self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.BIAS_output = np.array(
            [self.BiasOutputValue for i in range(self.OutputLayer)])
        self.classes_number = 2

    action_functions = {
        'sigmoid': (lambda x: 1 / (1 + np.exp(-x))),
        'tanh': (lambda x: np.tanh(x)),
        'Relu': (lambda x: x * (x > 0)),
    }
    derivative_functions = {
        'sigmoid': (lambda x: x * (1 - x)),
        'tanh': (lambda x: 1 - x ** 2),
        'Relu': (lambda x: 1 * (x > 0))
    }

    def starting_weights(self, x, y):
        return [[2 * random.random() - 1 for i in range(x)] for j in range(y)]

    def Backpropagation_Algorithm(self, x):
        '''
        Backpropagation Algorithm
        x: input (array)
        '''

        DELTA_output = []
        'Stage 1 - Error: OutputLayer'
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = ((-1) * (ERROR_output) * self.deriv(self.OUTPUT_L2))

        arrayStore = []
        'Stage 2 - Update weights OutputLayer and HiddenLayer'
        for i in range(self.hiddenLayer):
            for j in range(self.OutputLayer):
                self.WEIGHT_output[i][j] -= (self.learningRate *
                                             (DELTA_output[j] * self.OUTPUT_L1[i]))
                self.BIAS_output[j] -= (self.learningRate * DELTA_output[j])

        'Stage 3 - Error: HiddenLayer'
        delta_hidden = np.matmul(
            self.WEIGHT_output, DELTA_output) * self.deriv(self.OUTPUT_L1)

        'Stage 4 - Update weights HiddenLayer and InputLayer(x)'
        for i in range(self.OutputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= (self.learningRate *
                                             (delta_hidden[j] * x[i]))
                self.BIAS_hidden[j] -= (self.learningRate * delta_hidden[j])
