
# coding: utf-8

# In[17]:


import numpy as np
import csv
import psycopg2
import math
import pandas as pd


# In[18]:


df = pd.read_csv("data1.csv")


# In[19]:


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


# In[20]:


def mylinridgeregeval(X,w):
    yi = np.dot(X,w)
    return yi


# In[21]:


def meansquarederr(label_data,y):
    err = np.square((label_data) - (y))
    
#     print (err)
    err1 = np.average(err)
    return err1


# In[22]:


input_data = np.asarray(df.drop('x11', axis= 1))
label_data = np.asarray(df['x11'])
input_data=np.insert(input_data,0,np.ones(input_data.shape[0]),1)

label_data = np.reshape(label_data,(-1,1))
print (input_data.shape, label_data.shape)


# In[23]:


w = mylinridgereg(input_data,label_data,2)
yi = mylinridgeregeval(input_data,w)
yi = np.round(yi)

err=meansquarederr(label_data,yi)
print (err)

