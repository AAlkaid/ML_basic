#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv


# In[3]:


# load data
Dtree = open(r'AllElectronics.csv','r')
reader = csv.reader(Dtree)

# get the first line data
headers = reader.__next__()
print(headers)

# define two lists
featureList = []
labelList = []

# get the two lists
for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
    
print(featureList)


# In[7]:


# trans the data to 01
vec = DictVectorizer()
x_data = vec.fit_transform(featureList).toarray()

# print the feature names
print(vec.get_feature_names())

# print the label
print("LabelList: " + str(labelList))

# trans the label to 01
lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(labelList)
print("y_data: " + str(y_data))


# In[8]:


# create the model
model = tree.DecisionTreeClassifier(criterion='entropy')
# fit
model.fit(x_data, y_data)


# In[9]:


# test
x_test = x_data[0]
print("x_test: " + str(x_test))


# In[10]:


predict = model.predict(x_test.reshape(1,-1))
print("predict: " + str(predict))


# In[ ]:




