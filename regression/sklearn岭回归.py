#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[3]:


data = np.genfromtxt(r'longley.csv',delimiter=',')
print(data)


# In[12]:


x_data = data[1:,2:]
y_data = data[1:,1]
print(x_data)
print(y_data)


# In[13]:


# 生成50个值 (lamda)
alphas_to_test = np.linspace(0.001,1)

# create model
# 岭回归➕交叉验证（CV）
# alphas 岭回归系数，测试一下0.001-1中哪个值比较好，
# 然后保存store交叉验证后的结果，True就是保存结果
model = linear_model.RidgeCV(alphas=alphas_to_test, store_cv_values=True)
model.fit(x_data, y_data)

# print一下岭系数
print(model.alpha_)

# loss value
# cv是做了交叉验证法loss的值，（16，50），
# 16代表样本共16个，每次取1个作为测试集，其余15个作为训练集，每次得到一个loss
# 因为会做16次，所以有16个loss的值，50个岭系数的值，挨个对应的。
print(model.cv_values_.shape)


# In[14]:


# 画图
# 岭系数和loss值的关系
plt.plot(alphas_to_test, model.cv_values_.mean(axis=0))

plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)),'ro')
plt.show()


# In[16]:


# 转换一下格式
model.predict(x_data[2,np.newaxis])


# In[ ]:


plt.plot(x_data, model.predict(x_data))

