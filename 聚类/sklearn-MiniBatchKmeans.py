#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


# In[2]:


# load data
data = np.genfromtxt('kmeans.txt', delimiter=' ')

# set k value
k = 4


# In[3]:


model = MiniBatchKMeans(n_clusters=k)
model.fit(data)


# In[4]:


centers = model.cluster_centers_


# In[5]:


centers


# In[ ]:


result = mode.

