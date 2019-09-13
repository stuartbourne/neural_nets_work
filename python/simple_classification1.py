import numpy as np
from matplotlib import pyplot as plt

feature_set = np.array([[0, 1], [0, 0], [1, 0], [1, 1], [1, 1]])
plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:,0], feature_set[:, 1])
labels = np.array([1, 0, 0, 1, 1])
labels = labels.reshape(5, 1)
# plt.plot(labels)
plt.show()
#lets generate random weights and biases for the start
np.random.seed(42)
#weights = np.random.rand(3, 1)
weights = np.array([0.23, 0.88])
weights = weights.reshape(2, 1)
#bias = np.random.rand(1)
bias = 2
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

#l now lets actually train
for epoch in range(2000):
    inputs = feature_set

    #feedforward step 1
    XW = np.dot(feature_set, weights) + bias

    #feedforward step 2
    z = sigmoid(XW)

    #backprop step 1
    error = z - labels

    print("error: ", error.sum())
    #backprop step 2
    dcost_dpred =error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz
    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num

single_point = np.array([0, 1])
result = sigmoid(np.dot(single_point, weights) + bias)
print (result)
