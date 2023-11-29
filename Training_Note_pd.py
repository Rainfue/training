#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[19]:


df = pd.read_csv('https://raw.githubusercontent.com/jorisvandenbossche/pandas-tutorial/master/data/titanic.csv')


# In[20]:


pd.DataFrame({'Yes': [45, 34, 65], 'No': [54, 45, 23]})


# In[21]:


pd.Series([1, 2, 3, 4, 5, 6])


# In[22]:


df


# In[23]:


df.to_csv('./tmp.csv')


# In[24]:


df.info()


# In[26]:


df.shape


# In[27]:


df.columns


# In[28]:


df.head(8)


# In[29]:


df.tail(4)


# In[30]:


df.dtypes


# In[31]:


df['Name']


# In[32]:


type(df['Name'])


# In[33]:


df['Name'].shape


# In[34]:


df[['Name', 'Age']].head(3)


# In[35]:


df.loc[[5, 10, 15], ['Name', 'Age']]


# In[36]:


df.iloc[[5, 10, 15], [0, 1]]


# In[37]:


df.iloc[5:10, :3]


# In[38]:


df[df['Age']>18]


# In[39]:


df['Age']>18


# In[41]:


df[df['Age'].isin([5, 10, 15])]


# In[42]:


df[(df['Age'] == 5) | (df['Age'] == 10)] 


# In[44]:


df['Age'].notna()


# In[46]:


df[df['Age'].notna()]


# In[48]:


df['Age'].isna().sum()


# In[49]:


df.loc[df['Age'].notna(), 'Name']


# In[50]:


df.sort_values('Age').head(10)


# In[51]:


df.sort_values(['Age', 'Name'], ascending = [False, True]).head(10)


# In[52]:


df2 = df.copy(deep = True)


# In[54]:


cdf1 = pd.concat([df, df2])


# In[55]:


cdf1.shape


# In[56]:


cdf2 = pd.concat([df, df2], axis = 1)


# In[57]:


cdf2.shape


# In[59]:


mdf = pd.DataFrame(index = df.index)
mdf['PassengerId'] = df['PassengerId']
mdf['EvenId'] = mdf['PassengerId'].apply(lambda x: x % 2 == 0)


# In[60]:


mdf


# In[61]:


pd.merge(df, mdf, how = 'inner')


# In[63]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


iris = load_iris()


# In[65]:


type(iris)


# In[66]:


iris.data


# In[67]:


iris.feature_names


# In[68]:


iris.target_names


# In[69]:


iris.data.shape


# In[84]:


X = iris.data
y = iris.target


# In[85]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[86]:


k_range = range(1, 11)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))


# In[87]:


scores


# In[88]:


plt.plot(k_range, scores_list)
plt.xlabel('Number of k')
plt.ylabel('Accuracy')


# In[89]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y)


# In[90]:


classes = {0:'setosa', 1:'versicolor', 2:'virginica'}


# In[93]:


x_new = [[3, 4, 5, 2],
            [5, 4, 2, 2]]


# In[94]:


y_predict = knn.predict(x_new)


# In[ ]:


print(classes[y_[predict[o]]])
print(classes[y_[predict[1]]])

