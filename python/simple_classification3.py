#handles multi-class classification
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#generate random data sets
cat_images = np.random.randn(700, 2) + np.array([0, -3])
mouse_images = np.random.randn(700, 2) + np.array([3, 3])
dog_images = np.random.randn(700, 2) + np.array([-3, 3])

feature_set = np.vstack([cat_images, mouse_images, dog_images])

labels = np.array([0] * 700 + [1] * 700 + [2] * 700)
# we have now created an array of 2100 elements with the first 700 labelled as 0, the next as 1, and the final as 2
# these will act as our training data sets
# but we want one-hot encoded values, so lets convert
one_hot_labels = np.zeros((2100, 3))

for i in range(2100):
    one_hot_labels[i, labels[i]] = 1

plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels, cmap="plasma", s=30, alpha=0.5)
plt.show()

#now define softmax activation function for multi classification problems.
def softmax(A):
    expA = np.exp(A)
    return expA/expA.sum(axis=1, keepdims=True)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_d1(x):
    return sigmoid(x) * (1 - sigmoid(x))

instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 4
output_labels = 3

wh = np.random.rand(attributes, hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes, output_labels)
bo = np.random.rand(output_labels)
lr = 10e-4

error_cost = []

for epoch in range(50000):
#### feedforward

    #Phase 1
    zh = np.dot(feature_set, wh) + bh
    ah = sigmoid(zh)

    #Phase 2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

#### backprop

    #Phase 1
    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah
    dcost_dwo = np.dot(dzo_dwo.T, dcost_dzo)
    #now for the bias
    dcost_dbo = dcost_dzo

    #Phase 2
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
    dah_dzh = sigmoid_d1(zh)
    dzh_dwh = feature_set
    dcost_dwh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
    #now for bias
    dcost_dbh = dcost_dah * dah_dzh

#weight adjustment
    wo -= lr * dcost_dwo
    bo -= lr * dcost_dbo.sum(axis=0)
    wh -= lr * dcost_dwh
    bh -= lr * dcost_dbh.sum(axis=0)

    if epoch % 200 == 0:
        loss = np.sum(-one_hot_labels * np.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)
    