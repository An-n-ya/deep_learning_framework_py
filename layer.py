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

def main():
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
    plt.show()


main()