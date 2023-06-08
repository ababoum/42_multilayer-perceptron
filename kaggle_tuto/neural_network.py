import numpy as np


class NeuralNetwork():
    def __init__(self, num_layers=3, layers_sizes=[30, 15, 2]) -> None:
        self.layers = []
        self.num_layers = num_layers

        if (len(layers_sizes) != num_layers):
            raise Exception(
                'The number of layers is not equal to the number of sizes')

        for i in range(num_layers):
            self.layers.append(np.array([0] * layers_sizes[i]).reshape(-1, 1))

    def __getitem__(self, index):
        return self.layers[index]
