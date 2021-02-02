#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


data = np.genfromtxt("LR-testSet2.txt", delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]

plt.scatter(x_data[:,0], x_data[:,1], c=y_data)
plt.show()


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)


# In[23]:


def plot(model):
    x_min, x_max = x_data[:,0].min() - 1, x_data[:,0].max() + 1
    y_min, y_max = x_data[:,1].min() - 1, x_data[:,1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max,0.02),
                         np.arange(y_min, y_max,0.02))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape )
    
    cs = plt.contourf(xx, yy, z)
    
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test)
    plt.show()


# In[24]:


dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)
plot(dtree)
dtree.score(x_test, y_test)


# In[25]:


RF = RandomForestClassifier()
RF.fit(x_train, y_train)
plot(RF)
RF.score(x_test, y_test)


# In[ ]:




