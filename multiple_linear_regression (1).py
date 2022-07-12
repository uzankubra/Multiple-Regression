#!/usr/bin/env python
# coding: utf-8

# ### MULTIPLE LINEAR REGRESSION
# 
# Bir tane bağımlı değişken ile bununla ilişkisi olan bir dizi bağımsız değişken arasındaki ilişkiyi ortaya koymak için yapılan analizdir. Çoklu doğrusal regresyon iki ve daha fazla bağımsız değişken ve bir bağımlı değişken arasındaki doğrusal bağıntıyı inceler.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


df = pd.read_csv("multilinearregression.csv",sep = ";")


# In[3]:


df


# In[3]:


df[['alan', 'odasayisi', 'binayasi']]


# In[4]:


df['fiyat']


# In[4]:


reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

reg.predict([[230,4,10]])


# In[5]:


reg.predict([[230,6,0]])


# In[6]:


reg.predict([[355,3,20]])


# In[7]:


reg.predict([[230,4,10], [230,6,0], [355,3,20]])


# In[8]:


reg.coef_  #b1


# In[9]:


reg.intercept_  # a


# In[15]:


# y= a + b1X1 + b2X2 + b3X3 + ...

a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3

y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




