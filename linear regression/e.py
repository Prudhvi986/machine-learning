
# coding: utf-8

# In[18]:


import numpy as np
import csv
import psycopg2
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[19]:


df = pd.read_csv("data1.csv")


# In[20]:


def mylinridgereg(X,Y,lamda):
    p = (np.asmatrix(np.dot(X.T,X) + lamda*np.eye(len(X[0]))))
#     print(p.shape)
    p = np.linalg.pinv(p)
#     print(p.shape)
    q = np.dot(p,X.T)
#     print(q.shape)
    q = np.dot(q,Y)
#     print(Y.shape)
#     print(q.shape)
    return q
    


# In[21]:


def mylinridgeregeval(X,w):
    yi = np.dot(X,w)
    return yi


# In[22]:


def meansquarederr(label_data,y):
    err = np.square((label_data) - (y))
    
#     print (err)
    err1 = np.average(err)
    return err1


# In[23]:


input_data = np.asarray(df.drop('x11', axis= 1))
label_data = np.asarray(df['x11'])
input_data=np.insert(input_data,0,np.ones(input_data.shape[0]),1)

label_data = np.reshape(label_data,(-1,1))
print (input_data.shape, label_data.shape)


# In[24]:


features_train, features_test, target_train, target_test = train_test_split(input_data,label_data, test_size = 0.20, random_state = 10)


# In[25]:


lamdas = [1,1.5,2,2.5,3]
errl=[]
for i in range(len(lamdas)):
    w=mylinridgereg(features_train,target_train,lamdas[i])
    predicted_train = mylinridgeregeval(features_train,w)
    predicted_test = mylinridgeregeval(features_test,w)
    err = meansquarederr(target_test,predicted_test)
    errl.append(err)
    
lamda = lamdas[errl.index(min(errl))]
print (lamda)
weights = mylinridgereg(features_train,target_train,lamda)
y1 = mylinridgeregeval(features_test,weights)
err = meansquarederr(target_test,y1)
print (err)
    
     


# In[26]:


# w = mylinridgereg(input_data,label_data,2)
# yi = mylinridgeregeval(input_data,w)
# yi = np.round(yi)

# err=meansquarederr(label_data,yi)
# print (err)


# In[27]:


df = df.drop('x4',axis=1)
df = df.drop('x6',axis=1)


# In[28]:


input_data1 = np.asarray(df)


# In[29]:


features_train, features_test, target_train, target_test = train_test_split(input_data1,label_data, test_size = 0.20, random_state = 10)


# In[30]:


weights = mylinridgereg(features_train,target_train,lamda)
y1 = mylinridgeregeval(features_test,weights)
err = meansquarederr(target_test,y1)
print (err)


# In[31]:


df = df.drop('x5',axis=1)


# In[32]:


input_data2 = np.asarray(df)


# In[33]:


features_train, features_test, target_train, target_test = train_test_split(input_data2,label_data, test_size = 0.20, random_state = 10)


# In[34]:


weights = mylinridgereg(features_train,target_train,lamda)
y1 = mylinridgeregeval(features_test,weights)
err = meansquarederr(target_test,y1)
print (err)

