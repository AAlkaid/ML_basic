#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt


# In[22]:


X = np.array([[1,0,0,0,0,0],
              [1,0,1,0,0,1],
              [1,1,0,1,0,0],
              [1,0,1,1,1,1]])

# label
Y = np.array([-1,1,1,-1])

# init weight
W = (np.random.random(6) - 0.5)*2

# learning rate
lr = 0.11
n = 0
# Output
O = 0


# In[31]:


def update():
    global X,Y,W,lr,n
    n += 1
    O = np.dot(X,W.T)
    W_C = lr*((Y-O.T).dot(X)) / int(X.shape[0])
    W = W + W_C


# In[32]:


for i in range(10000):
    # 更新一次，打印一次
    update()

    # 0.1 0.1 0.2 -0.2
    #  1   1   1   -1
# positive sample
x1 = [0,1]
y1 = [1,0]

# negative sample
x2 = [0,1]
y2 = [0,1]

def calculate(x, root):
    a = W[5]
    b = W[2] + x*W[4]
    c = W[0] + x*W[1] + x*x*W[3]
    
    if root == 1:
        return (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
    if root == 2:
        return (-b - np.sqrt(b*b - 4*a*c)) / (2*a)


xdata = np.linspace(-1,3)

plt.figure()

plt.plot(xdata,calculate(xdata,1),'r')
plt.plot(xdata,calculate(xdata,2),'r')

plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'yo')
plt.show()


# In[ ]:





# In[ ]:




