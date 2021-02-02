#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import classification_report


# In[3]:


# get random data
x1, y1 = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
x2, y2 = make_gaussian_quantiles(mean=(3,3),n_samples=500,n_features=2,n_classes=2)

# mix the datasets
x_data = np.concatenate((x1,x2))
y_data = np.concatenate((y1, -y2+1))


# In[4]:


plt.scatter(x_data[:,0], x_data[:,1], c=y_data)


# In[7]:


model = tree.DecisionTreeClassifier(max_depth=3)

model.fit(x_data, y_data)

x_min, x_max = x_data[:,0].min() - 1, x_data[:,0].max() + 1
y_min, y_max = x_data[:,1].min() - 1, x_data[:,1].max() + 1


xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02)
                    ,np.arange(y_min, y_max, 0.02))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# 等高线
cs = plt.contourf(xx, yy, z)

plt.scatter(x_data[:,0], x_data[:,1], c=y_data)
plt.show()


# In[8]:


model.score(x_data, y_data)


# In[ ]:


model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=10)

model.fit(x_data, y_data)

