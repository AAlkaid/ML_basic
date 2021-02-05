#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# loda data
data = np.genfromtxt('kmeans.txt',delimiter=' ')
# number of clusters
k = 4


# In[4]:


# train model

model = KMeans(n_clusters=k)
model.fit(data)


# In[5]:


centers = model.cluster_centers_
print(centers)


# In[6]:


result = model.predict(data)
print(result)


# In[7]:


model.labels_


# In[ ]:


mark = ['or', 'ob', 'og', 'oy']

for i,d in enumerate(data):
    plt.plot(d[0], d[1], mark[result[i]])
    
mark = ['*r', '*b', '*g', '*y']
for i, center in enumerate(centers):
    plt.plot(center[0],)

