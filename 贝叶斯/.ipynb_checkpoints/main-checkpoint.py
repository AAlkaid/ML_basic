from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

news = fetch_20newsgroups(subset='all')

print(news.target_names)
print(len(news.data))
print(len(news.target))


print(news.target[0])
print(news.target_names[news.target[0]])



x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)

from sklearn.feature_extraction.text import CountVectorizer

texts = ['dog cat fish',
         'dog cat cat',
         'fish bird',
          'bird']

cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)

print(cv.get_feature_names())
print(cv_fit.toarray())

print(cv_fit.toarray().sum(axis=0))

