#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[9]:


# load data
data = np.genfromtxt('job.csv', delimiter=',')
x_data = data[1:,1]
y_data = data[1:,2]
plt.scatter(x_data,y_data)
plt.show()


# In[10]:


#x_data = data[1:,1,np.newaxis]
#y_data = data[1:,2,np.newaxis]

x_data = x_data[:,np.newaxis]
y_data = y_data[:,np.newaxis]

model = LinearRegression()
model.fit(x_data, y_data)


# In[11]:


# draw picture
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data),'r')
plt.show()


# In[19]:


# 多项式回归
poly_reg = PolynomialFeatures(degree=5)
# 特征处理
x_poly = poly_reg.fit_transform(x_data)

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_data)


# In[20]:


# draw the picture
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)),c='r')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show


# In[ ]:




