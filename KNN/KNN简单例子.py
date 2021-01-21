#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import operator


# In[3]:


# 已知的数据
x1 = np.array([3,2,1])
y1 = np.array([104,100,81])
x2 = np.array([101,99,98])
y2 = np.array([10,5,2])


scatter1 = plt.scatter(x1,y1,c='r')
scatter2 = plt.scatter(x2,y2,c='b')

# 未知数据
x = np.array([18])
y = np.array([90])
scatter3 = plt.scatter(x,y,c='k')

# plot
plt.legend(handles=[scatter1,scatter2,scatter3],labels=['labelA','labelB','X'],loc='best')

plt.show()


# In[4]:


x_data = np.array([[3,104],
                   [2,100],
                   [1,81],
                   [101,10],
                   [99,5],
                   [81,2]])
y_data = np.array(['A','A','A','B','B','B'])

x_test = np.array([18,90])


# In[5]:


x_data_size = x_data.shape[0]
x_data_size


# In[6]:


# copy x_test
np.tile(x_test, (x_data_size,1))


# In[8]:


# 计算差值
diffMat = np.tile(x_test, (x_data_size,1)) - x_data
diffMat


# In[9]:


# 计算差值的平方
sqDiffMat = diffMat**2
sqDiffMat


# In[10]:


# calculate the sum
# each row
sqDistance = sqDiffMat.sum(axis=1)
sqDistance


# In[11]:


# sqrt
distances = sqDistance**0.5
distances


# In[12]:


# sort the index
sortDistances = distances.argsort()
sortDistances


# In[13]:


classCount = {}
# set K
# the closest 5 points

k = 5
for i in range(k):
    # get labels
    votelabel = y_data[sortDistances[i]]
    
    # get the numbers of labels
    classCount[votelabel] = classCount.get(votelabel, 0) + 1


# In[14]:


classCount


# In[16]:


sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
sortedClassCount


# In[17]:


# get the most label
knnclass = sortedClassCount[0][0]
knnclass


# In[ ]:




