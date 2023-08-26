#!/usr/bin/env python
# coding: utf-8

# In[34]:


#library
# Importing some Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


import warnings 
warnings.filterwarnings('ignore')


# In[36]:


#data set Requirements

df = pd.read_csv("Iris.csv")


# In[37]:


#Visualization

df.head()


# In[38]:


df.shape


# In[39]:


#Data Frame Descriptions

df.describe()


# In[40]:


#Data Frame Information given

df.info


# In[41]:


#Data Explooration

df.Species.value_counts


# In[42]:


#Data Pre-processing

x = df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']


# In[43]:


#Here x= variable
x


# In[44]:


#here y = Variable
y


# In[45]:


#Data Spliting

import sklearn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)
#Linear Regression model 
#zRabdom Forest model


# In[46]:


from sklearn.linear_model import LogisticRegression


# In[1]:


#here we use LgisticRegression model
lr = LogisticRegression()

#model training starts
lr.fit(x,y)

#traning sets listing 
lr.fit(x_train, y_train)


# In[48]:


#getting predictions
Prediction = lr.predict(x)


# In[49]:


#comparing between "actual" and "prediction"
Scores = pd.DataFrame({"Actual":y,"Prediction":Prediction})
Scores.head


# In[50]:


y_test_hat = lr.predict(x_test)


# In[51]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat)*100,'%')


# Thank you!!!

# In[ ]:





# In[ ]:





# In[ ]:




