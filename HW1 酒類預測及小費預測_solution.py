#!/usr/bin/env python
# coding: utf-8

# # Lab1 酒類分類

# ## 1. 載入資料集 

# In[1]:


# 載入套件
import numpy as np
import pandas as pd
from sklearn import datasets

# 載入資料集
ds = datasets.load_wine()
print(ds.DESCR)


# In[2]:


ds.data


# In[3]:


#取得資料集的資料
data = ds.data
df = pd.DataFrame(data, columns=ds.feature_names)
df


# In[4]:


X = df
y = ds.target
print('取前10筆資料')
print(df.head(10))


# In[5]:


print('取後10筆資料')
print(df.tail(10))


# In[6]:


print('採樣10筆資料')
print(df.sample(10))


# In[7]:


# 類別名稱
print(ds.target_names)


# In[8]:


df.shape


# In[9]:


print('此DataFrame的詳細資訊')
print(df.info())


# In[10]:


# missing value
print('檢查資料中是否有null值')
df.isna().sum()


# In[11]:


# 描述統計量
df.describe()


# ## 2/3 省略
# ## 4. 資料分割

# In[12]:


# 資料分割為訓練及測試資料
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## 5. 選擇演算法

# In[13]:


from sklearn.neighbors import KNeighborsClassifier

# 選擇演算法
# 超參數(hyperparameter)：n_neighbors=3
# KNeighborsClassifier： 簡稱KNN
clf = KNeighborsClassifier(n_neighbors=3)


# ## 6. 模型訓練

# In[14]:


# 模型訓練
clf.fit(X_train, y_train)


# ## 7. 打分數

# In[15]:


# 打分數
score = clf.score(X_test, y_test)
print(f'score = {score}')


# ## 8. 評估
# ## LogisticRegression

# In[16]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f'score = {score}')


# In[38]:


from sklearn.metrics import confusion_matrix,accuracy_score

accuracy_score(y_test,clf.predict(X_test))


# In[17]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, clf.predict(X_test))


# In[ ]:


confusion_matrix(y_test,clf.predict(X_test)) #不在對角線就是錯誤


# ## 9. 佈署：存檔/載入模型

# In[18]:


# 存檔
from joblib import dump, load

dump(clf, 'model1.joblib') 


# In[19]:


# 載入模型
clf = load('model1.joblib') 


# ## 10. 預測

# ### 採用隨機亂數生成資料

# In[24]:


# 採用隨機亂數生成資料，預測酒類品種
data = []
for i in range(df.shape[1]):
    data.extend(np.random.normal(df[df.columns[i]].mean(), df[df.columns[i]].std(), 1))#產生1筆
    
data


# In[25]:


clf.predict([data])


# In[26]:


# 預測機率
print(clf.predict_proba([data])) #y是0的機率0.00187952 1的機率0.99376398 2的機率0.0043565


# ### 在現成的資料集抽樣

# In[27]:


data = df.sample(3)
data


# In[28]:


data.index


# In[29]:


y


# In[30]:


# 真實的結果
y[data.index]


# In[31]:


# 預測
clf.predict(data)


# In[32]:


# 比對
y[data.index] == clf.predict(data)


# # Lab2 小費預測

# In[40]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[43]:


df = pd.read_csv('./tips.csv')#.代表目前目錄
df.head()
#tip為y


# In[44]:


X = df.drop('tip', axis=1)
X.head()


# In[45]:


y = df['tip']
y.head()


# In[46]:


X['sex'].unique()


# In[47]:


X['sex'].map({'Female':0, 'Male':1})


# In[48]:


X['sex'] = X['sex'].map({'Female':0, 'Male':1})
X.head()


# In[49]:


X['smoker'].unique()


# In[50]:


X['smoker'] = X['smoker'].map({'No':0, 'Yes':1})
X.head()


# In[51]:


X['day'].unique()


# In[52]:


X['day'] = X['day'].map({'Sun':0, 'Sat':1, 'Thur':2, 'Fri':3})
X.head()


# In[53]:


X['time'].unique()


# In[54]:


X['time'] = X['time'].map({'Dinner':0, 'Lunch':1})
X.head()


# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape, X_test.shape


# In[56]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize=True)#特徵縮放


# In[57]:


lr.fit(X_train, y_train)


# In[58]:


lr.coef_


# In[59]:


lr.intercept_


# In[60]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = lr.predict(X_test)
y_pred


# In[61]:


mean_absolute_error(y_test, y_pred)


# In[62]:


mean_squared_error(y_test, y_pred)


# In[63]:


r2_score(y_test, y_pred) #0~1


# In[79]:


x1 = X.sample()
print(lr.predict(x1))
print()


# In[80]:


x1


# In[81]:


y[x1.index]


# In[78]:


x1 = [[50, 1, 0, 1, 1, 4], [5, 1, 0, 1, 1, 4]]
print(lr.predict(x1))


# In[82]:


from joblib import dump
dump(lr,'tips.joblib')


# ### seaborn

# In[84]:


import seaborn  as sns
import pandas as pd


# In[85]:


df = sns.load_dataset('tips')


# In[86]:


df.head()


# In[87]:


df = sns.load_dataset('titanic')


# In[89]:


df.head()


# In[ ]:




