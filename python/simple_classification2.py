from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

#lets generate some sample data to work with
np.random.seed(0)
feature_set, labels1 = datasets.make_moons(100, noise=0.1)
plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels1, cmap=plt.cm.winter)
x_test = np.arange(-2, 3, step=5/len(feature_set))
y_test = np.arange(-2, 3, step=5/len(feature_set))
plt.plot(x_test, y_test)

labels = labels1.reshape(100, 1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_d1(x):
    return sigmoid(x) * (1 - sigmoid(x))

#weights to the hidden layer
wh = np.random.rand(len(feature_set[0]), 4)
#weights to the output layer
wo = np.random.rand(4, 1)
lr = 0.01

def feed_forward(x_test, y_test, wh, wo):
    ao_list = []
    i = 0
    while i < len(x_test):    
        zh = np.dot([x_test[i], y_test[i]], wh)
        ah = sigmoid(zh)
        zo = np.dot(ah, wo)
        ao = sigmoid(zo)
        ao_list.append(ao)
        i = i + 1
    
    return ao_list

for epoch in range(200000):
    #feedforward
    zh = np.dot(feature_set, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    #now for the fun backprop stuff
    #phase 1 ----------------------
    error_out = ((1/2)* (np.power(ao - labels, 2)))
    print(error_out.sum())

    dcost_dao = ao - labels
    dao_dzo = sigmoid_d1(ao)
    dzo_dwo = ah

    dcost_dwo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    #phase 2 ----------------------
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo, dzo_dah.T)

    dah_dzh = sigmoid_d1(zh)
    dzh_dwh = feature_set
    dcost_dwh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    #now we update the weights
    wh -= lr * dcost_dwh
    wo -= lr * dcost_dwo
    plt.pause(0.05)
    plt.cla()
    plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels1, cmap=plt.cm.winter)
    y_test = np.array(feed_forward(x_test, y_test, wh, wo))
    plt.plot(x_test, y_test)

single_point_x = np.array([0.4, 0.4])
single_point_y = np.array([0.4, 0.5])
ao = feed_forward(single_point_x, single_point_y, wh, wo)
print("out: ", ao)
plt.show()
