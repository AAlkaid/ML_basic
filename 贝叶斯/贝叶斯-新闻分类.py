#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


text = ['The quick brown fox jumped over the lazy dog.',
        'The dog',
        'The fox']

vectorizer = TfidfVectorizer()

vectorizer.fit(text)


print(vectorizer.vocabulary_)
print(vectorizer.idf_)


# In[3]:


vector = vectorizer.transform([text[0]])
print(vector.shape)
print(vector.toarray())


# In[ ]:




