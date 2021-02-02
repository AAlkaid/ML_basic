#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np


# In[22]:


iris = datasets.load_iris()

x_data, y_data = iris.data[:,1:3], iris.target
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()


# 次级分类器
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr)

for clf,label in zip([clf1,clf2,clf3,sclf],
                    ['KNN','Decison Tree','LogisticRegression','StackClassfier']):
    scores = model_selection.cross_val_score(clf, x_data, y_data, cv=3, scoring='accuracy')
    print("Acc : %0.2f [%s]" % (scores.mean(), label))


# In[ ]:




