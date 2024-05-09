from nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

def main():
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
    plt.show()


main()