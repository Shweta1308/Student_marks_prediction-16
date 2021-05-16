#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir("E:\Shweta Coding\kaggle datasets")


# In[3]:


data = pd.read_csv("student_info.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


plt.scatter(x= data.study_hours, y = data.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students Marks")
plt.title("Scatter Plot Of Students Study Hours vs Students Marks")
plt.show()


# In[9]:


data.isnull().sum()


# In[10]:


data.mean()


# In[11]:


data2 = data.fillna(data.mean())
data2


# In[12]:


data2.isnull().sum()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X = data2.drop('student_marks',axis=1)
y = data['student_marks']

print("shape of X =",X.shape)
print("shape of y =",y.shape)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=51)
print("shape of X_train =", X_train.shape)
print("shape of y_train =",y_train.shape)
print("shape of X_test =",X_test.shape)
print("shape of y_test =",y_test.shape)


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


linear_model = LinearRegression()
linear_model.fit(X_train,y_train)


# In[18]:


linear_model.coef_


# In[19]:


linear_model.intercept_


# In[20]:


linear_model.predict([[5]])[0].round(2)


# In[21]:


y_pred = linear_model.predict(X_test)
y_pred


# In[22]:


y_test


# In[23]:


data = pd.DataFrame(np.c_[X_test,y_test,y_pred],columns=["study_hours","student_marks_original","student_marks_predicted"])
data


# In[24]:


linear_model.score(X_test, y_test)


# In[25]:


plt.scatter(X_test,y_test)


# In[26]:


plt.scatter(X_test,y_test)
plt.plot(X_train,linear_model.predict(X_train),color='r')


# In[27]:


import pickle


# In[28]:


pickle.dump(linear_model,open("reg_model.pkl",'wb'))


# In[29]:


model = pickle.load(open("reg_model.pkl",'rb'))


# In[30]:


linear_model.predict([[5]])[0].round(2)


# In[ ]:




