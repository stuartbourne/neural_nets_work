#lets generate some sample data to work with
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.1)
plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels, cmap=plt.cm.winter)
plt.show()