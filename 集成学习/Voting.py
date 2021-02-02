#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np


# In[2]:


iris = datasets.load_iris()

x_data, y_data = iris.data[:,1:3], iris.target
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()


sclf = VotingClassifier([('KNN',clf1),('dtree',clf2),('lr',clf3)])

for clf,label in zip([clf1,clf2,clf3,sclf],
                    ['KNN','Decison Tree','LogisticRegression','StackClassfier']):
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=3, scoring='accuracy')
    print("Acc : %0.2f [%s]" % (scores.mean(), label))


# In[ ]:




