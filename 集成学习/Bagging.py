#!/usr/bin/env python
# coding: utf-8

# In[25]:


from sklearn import neighbors
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# In[26]:


iris = datasets.load_iris()
x_data = iris.data[:, :2]
y_data = iris.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)


# In[27]:


knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)


# In[28]:


def plot(model):
    x_min, x_max = x_data[:,0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:,1].min() - 1, x_data[:, 1].max() + 1
    
    # 网格矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    cs = plt.contourf(xx, yy, z)


# In[29]:


plot(knn)

plt.scatter(x_data[:,0], x_data[:,1], c=y_data)
plt.show()

knn.score(x_test, y_test)


# In[31]:


dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)


# In[32]:


plot(dtree)

plt.scatter(x_data[:,0], x_data[:,1], c=y_data)
plt.show()

dtree.score(x_test, y_test)


# In[33]:


# 又放回抽样100次
bagging_knn = BaggingClassifier(knn, n_estimators=100)

bagging_knn.fit(x_train, y_train)
plot(bagging_knn)

plt.scatter(x_data[:,0], x_data[:,1], c=y_data)
plt.show()
bagging_knn.score(x_test, y_test)


# In[21]:


plot(bagging_knn)


# In[34]:


# 又放回抽样100次
bagging_dtree = BaggingClassifier(dtree, n_estimators=100)

bagging_dtree.fit(x_train, y_train)
plot(bagging_dtree)

plt.scatter(x_data[:,0], x_data[:,1], c=y_data)
plt.show()
bagging_dtree.score(x_test, y_test)


# In[ ]:





# In[ ]:




