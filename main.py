import nnfs
from loss import Loss_CategoricalCrossEntropy
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data


nnfs.init()
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()
a = Loss_CategoricalCrossEntropy();