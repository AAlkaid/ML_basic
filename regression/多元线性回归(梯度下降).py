#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[36]:


# load data

data = genfromtxt(r'Delivery.csv',delimiter=',')
print(data)


# In[37]:


# split data
# x: the first and second raw
# y: the last raw
x_data = data[:,:-1]
y_data = data[:,-1]
print(x_data)
print(y_data)


# In[40]:


# learing rate
learning_rate = 0.0001

# parameters
theta0 = 0
theta1 = 0
theta2 = 0

# the maximum iteration times
epochs = 1000


#最小二乘法
def compute_error(theta0, theta1, theta2, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += ((theta0 + theta1 * x_data[i,0] + theta2 * x_data[i,1]) - y_data[i]) ** 2
    return totalError / float(len(x_data)) 

# 梯度下降
def gradient_descent_method(x_data, y_data, theta0, theta1, theta2, learning_rate, epochs):
    # total numeber of data
    
    m = float(len(x_data))
    
    # 循环epochs次
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        
        # 梯度总和求平均
        for j in range(0,len(x_data)):
            theta0_grad += (1/m) *      1      * ((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])
            theta1_grad += (1/m) * x_data[j,0] * ((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])
            theta2_grad += (1/m) * x_data[j,1] * ((theta0 + theta1*x_data[j,0] + theta2*x_data[j,1]) - y_data[j])
            
        # update k and b
        theta0 = theta0 - (learning_rate * theta0_grad)
        theta1 = theta1 - (learning_rate * theta1_grad)
        theta2 = theta2 - (learning_rate * theta2_grad)
    return theta0,theta1,theta2


# In[41]:


print('Starting  theta0 = {0}, theta1 = {1}, theta2 = {2}, error = {3}'
      .format(theta0, theta1, theta2, compute_error(theta0, theta1, theta2, x_data, y_data)))
print('Running Now! Pls wait...')

theta0, theta1, theta2 = gradient_descent_method(x_data, y_data, theta0, theta1, theta2, learning_rate, epochs)
print('After {0} iterations theta0 = {1}, theta1 = {2}, theta2 = {3}, error = {4}'
     .format(epochs, theta0, theta1, theta2, compute_error(theta0, theta1, theta2, x_data, y_data)))


# In[43]:


ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(x_data[:,0], x_data[:,1], y_data, c = 'r', marker = 'o',s = 100)
x0 = x_data[:,0]
x1 = x_data[:,1]

# 生成网格矩阵
x0,x1 = np.meshgrid(x0, x1)
z = theta0 + x0 * theta1 + x1 * theta2

# 画3D图像
ax.plot_surface(x0, x1, z)

# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')

# show figure
plt.show()


# In[ ]:





# In[ ]:




