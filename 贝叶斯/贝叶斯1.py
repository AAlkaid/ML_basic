#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB


# In[4]:


# load data
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)


# In[9]:


mul_nb = GaussianNB()
mul_nb.fit(x_train, y_train)


# In[10]:


print(classification_report(mul_nb.predict(x_test), y_test))


# In[11]:


print(confusion_matrix(mul_nb.predict(x_test), y_test))


# In[ ]:




