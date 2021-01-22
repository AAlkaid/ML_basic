#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[7]:


X = np.array([[1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])
Y = np.array([[0,1,1,0]])

V = np.random.random((3,4))*2 - 1
W = np.random.random((4,1))*2 - 1

print(V)
print(W)

learning_rate = 0.11

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def update():
    global X,Y,W,V,learning_rate
    
    L1 = sigmoid(np.dot(X,V)) # 隐藏层输出 4 x 4 matrix
    L2 = sigmoid(np.dot(L1,W)) # output layer 4x1 matrix
    
    L2_delta = (Y.T - L2) * dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)
    
    W_C = learning_rate*L1.T.dot(L2_delta)
    V_C = learning_rate*X.T.dot(L1_delta)
    
    W += W_C
    V += V_C


# In[ ]:





# In[10]:


for i in range(20000):
    update()
    
    if i % 500 == 0:
        L1 = sigmoid(np.dot(X,V))
        L2 = sigmoid(np.dot(L1,W))
        
        print("error : " ,np.mean(np.abs(Y.T - L2)))
L1 = sigmoid(np.dot(X,V))
L2 = sigmoid(np.dot(L1,W))
print(L2)


def judge(x):
    if x >= 0.5:
        return 1
    else:
        return 0
for i in map(judge,L2):
    print(i)


# In[ ]:




