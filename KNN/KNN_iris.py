#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import operator
import random


# In[16]:


def knn(x_test, x_data, y_data, k):
    # calculate the num of samples
    x_data_size = x_data.shape[0]
    # copy x_test
    np.tile(x_test, (x_data_size, 1))
    diffMat = np.tile(x_test, (x_data_size,1)) - x_data
    # calculate the square
    sqDiffMat = diffMat**2
    # calculate the sum
    sqDistances = sqDiffMat.sum(axis=1)
    # calculate the sqrt
    distances = sqDistances**0.5
    # sort from small to large
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        # get the labels
        votelabel = y_data[sortedDistances[i]]
        # calculate the numbers of the labels
        classCount[votelabel] = classCount.get(votelabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]


# In[19]:


iris = datasets.load_iris()
# mix the data
data_size = iris.data.shape[0]
index = [i for i in range(data_size)]

# mix the data automatically
random.shuffle(index)

# get the data 
# get the mixed data
iris.data = iris.data[index]
iris.target = iris.target[index]

# split the data
test_size = 40
x_train = iris.data[test_size:]
x_test = iris.data[:test_size]

y_train = iris.target[test_size:]
y_test = iris.target[:test_size]

predictions = []

# 40次
for i in range(x_test.shape[0]):
    predictions.append(knn(x_test[i], x_train, y_train, 5))
    
print(classification_report(y_test, predictions))


# In[21]:


print(confusion_matrix(y_test, predictions))


# In[ ]:




