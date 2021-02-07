#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# load data
data = np.genfromtxt('data.csv', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]

plt.scatter(x_data, y_data)
plt.show()


# In[6]:


# 中心化
def zeroMean(dataMat):
    # axis=0 按列求平均
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal


# In[12]:


newData, meanVal = zeroMean(data)

# calculate the cov
covMat = np.cov(newData, rowvar=0)


# In[13]:


covMat


# In[15]:


eigVals, eigVects = np.linalg.eig(np.mat(covMat))


# In[16]:


eigVals


# In[17]:


eigVects


# In[18]:


eigValIndice = np.argsort(eigVals)


# In[19]:


eigValIndice


# In[21]:


top = 1

n_eigValIndice = eigValIndice[-1:-(top+1):-1]


# In[22]:


n_eigValIndice


# In[25]:


n_eigVect = eigVects[:, n_eigValIndice]
n_eigVect


# In[26]:


lowDataMat = newData * n_eigVect


# In[27]:


lowDataMat


# In[29]:


# the new data
reconMat = (lowDataMat * n_eigVect.T) + meanVal


# In[30]:


reconMat


# In[31]:


# load data
data = np.genfromtxt('data.csv', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]
plt.scatter(x_data, y_data)



# new data
x_data = np.array(reconMat)[:, 0]
y_data = np.array(reconMat)[:, 1]
plt.scatter(x_data, y_data, c='r')

plt.show()


# In[ ]:




