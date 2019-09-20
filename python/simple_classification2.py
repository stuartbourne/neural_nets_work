from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import os.path
from math import exp

DATA_FILENAME="classification2_data.txt"
#lets generate some sample data to work with
np.random.seed(0)
feature_set, labels = datasets.make_moons(100, noise=0.1)
output_file = open(DATA_FILENAME, 'w+')
if os.path.exists(DATA_FILENAME):
    for i in range(len(feature_set)):
        data_line = "{0:f} {1:f} {2:f}\n".format(feature_set[i, 0],feature_set[i, 1],labels[i])
        output_file.write(data_line)
    output_file.close()
plt.figure(figsize=(10, 7))
plt.scatter(feature_set[:, 0], feature_set[:, 1], c=labels, cmap=plt.cm.winter)
plt.show()

labels = labels.reshape(100, 1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

wh = np.random.rand(len(feature_set[0]),4)
wo = np.random.rand(4, 1)
lr = 0.5

for epoch in range(20000):
    # feedforward
    zh = np.dot(feature_set, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # Phase1 =======================

    error_out = ((1 / 2) * (np.power((ao - labels), 2)))
    print(error_out.sum())

    dcost_dao = ao - labels
    dao_dzo = sigmoid_der(zo)
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # Phase 2 =======================

    # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # dcost_dah = dcost_dzo * dzo_dah
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    # Update Weights ================

    wh -= lr * dcost_wh
    wo -= lr * dcost_wo

#now lets test two points
test_feature = np.array([0, 1]) #should be a 0
zh = np.dot(test_feature, wh)
ah = sigmoid(zh)
zo = np.dot(ah, wo)
ao = sigmoid(zo)
print("Activation out from 0, 1 (should be 0): ", ao)

test_feature = np.array([1, -0.5]) #should be a 1
zh = np.dot(test_feature, wh)
ah = sigmoid(zh)
zo = np.dot(ah, wo)
ao = sigmoid(zo)
print("Activation out from 1, -0.5 (should be 1): ", ao)
