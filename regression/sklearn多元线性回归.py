#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[3]:


# load data
data = genfromtxt(r'Delivery.csv', delimiter=',')
print(data)


# In[4]:


x_data = data[:,:-1]
y_data = data[:,-1]
print(x_data)
print(y_data)


# In[5]:


# create the model
model = linear_model.LinearRegression()
model.fit(x_data, y_data)


# In[6]:


# 系数
print('coefficients:',model.coef_)

# 截距
print('intercept:',model.intercept_)


# In[7]:


# test
x_test = [[102,4]]
predict = model.predict(x_test)
print('predict:',predict)


# In[8]:


# draw the picture

ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(x_data[:,0], x_data[:,1], y_data, c='r', marker='o', s=100)

x0 = x_data[:,0]
x1 = x_data[:,1]


x0,x1 = np.meshgrid(x0,x1)
z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]

# draw 3D figure
ax.plot_surface(x0,x1,z)

# set axis

ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')

# show the figure
plt.show()


# In[ ]:




