#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# loading data
data = np.genfromtxt('data.csv',delimiter=',')
x_data = data[:,0]
y_data = data[:,1]

plt.scatter(x_data, y_data)
plt.show()
print(x_data.shape)


# In[5]:


x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]

print(x_data.shape)
print(y_data.shape)

# 实例化
# 创建并且拟合模型
model = LinearRegression()
model.fit(x_data, y_data)


# In[6]:


plt.plot(x_data, y_data,'b.')
plt.plot(x_data, model.predict(x_data),'r')
plt.show()


# In[ ]:




