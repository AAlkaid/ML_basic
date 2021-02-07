#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# In[2]:


data = np.genfromtxt('kmeans.txt', delimiter=' ')


# In[3]:


model = DBSCAN(eps=1, min_samples=4)
model.fit(data)


# In[4]:


result = model.fit_predict(data)
result


# In[5]:


mark = ['or', 'ob', 'og', 'oy', 'ok', 'om']
for i,d in enumerate(data):
    plt.plot(d[0], d[1], mark[result[i]])
    
plt.show()


# In[ ]:




