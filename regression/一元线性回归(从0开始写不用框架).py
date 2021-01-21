#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


# load data

data = np.genfromtxt('data.csv', delimiter=',') # 分隔符是逗号

# 切分数据，分为x和y
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data, y_data)
plt.show()


# In[19]:


# 学习率
learning_rate = 0.0001

# 斜率
k = 0

# 截距
b = 0

# 最大迭代次数
epochs = 50

# 最小二乘法算的误差
def compute_error(b, k, x_data, y_data):
    total_Error = 0
    for i in range(0, len(x_data)):
        total_Error += (y_data[i] - (k * x_data[i] + b)) ** 2
    return total_Error / float(len(x_data)) / 2.0


# 梯度下降
def gradien_descent_method(x_data, y_data, b, k, learning_rate, epochs):
    
    # 总样本个数
    m = float(len(x_data))
    
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        
        # calculate the sum of gradient
        
        for j in range(0, len(x_data)):
            # 两个偏导数
            b_grad += (1/m) * ((k * x_data[j] + b) - y_data[j])
            k_grad += (1/m) * x_data[j] * ((k * x_data[i] + b) - y_data[j])
            
        # 更新k和b
        b = b - (learning_rate * b_grad)
        k = k - (learning_rate * k_grad)
        
        # 每迭代5次，输出一次图像
        if i % 5 == 0:
            print('epochs:',i)
            plt.plot(x_data, y_data, 'b.')
            plt.plot(x_data, k * x_data + b, 'r')
            plt.show()
        
    return b, k


# In[20]:


print('Starting b = {0}, k = {1}, error = {2}'.format(b, k, compute_error(b, k, x_data, y_data)))
print('Running...')

b, k = gradien_descent_method(x_data, y_data, b, k ,learning_rate, epochs)
print('After {0} iterations b = {1}, k = {2}, error = {3}'.format(epochs, b, k, compute_error(b, k , x_data, y_data)))


# 画图
#plt.plot(x_data, y_data, 'b.')
#plt.plot(x_data, k * x_data + b, 'r')
#plt.show()


# In[ ]:





# In[ ]:




