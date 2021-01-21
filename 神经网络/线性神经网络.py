#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# input data
X = np.array([[1,3,3],
              [1,4,3],
              [1,1,1],
              [1,0,2]])

# labels
Y = np.array([[1],
              [1],
              [-1],
              [-1]])

# init weight
W = (np.random.random([3,1]) - 0.5)*2

# learning rate
lr = 0.11

# Output
O = 0


# In[3]:


def update():
    global X,Y,W,lr
    O = np.dot(X,W)
    W_C = lr*(X.T.dot(Y-O)) / int(X.shape[0])
    W = W + W_C


# In[5]:


for i in range(100):
    # 更新一次，打印一次
    update()
    print(W)
    print(i)
    
    O = np.dot(X,W)
    
    if (O == Y).all():
        print('finished')
        print('epoch:',i)
        break

# positive sample
x1 = [3,4]
y1 = [3,3]

# negative sample
x2 = [1,0]
y2 = [1,2]

# calculate k and b
k = -W[1]/W[2]
b = -W[0]/W[2]

print('k=',k)
print('b=',b)

xdata = (0,5)

plt.figure()
plt.plot(xdata,xdata*k+b,'r')
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()


# In[ ]:




