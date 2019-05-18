
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# In[22]:


def dBSCAN(xlist,eps,minpts):
    id=0
    for i in range(len(xlist)):
        if labels[i] == 0:
            neighbours = region(xlist,i,eps)
            if(len(neighbours)<minpts):
                labels[i] = -1
            else:
                id = id+1
                expandcluster(xlist,labels,i,neighbours,id,eps,minpts)
    return labels


# In[23]:


def expandcluster(xlist,labels,i,neighbours,id,eps,minpts):
    labels[i] = id
    j=0
    while(j< len(neighbours)):
        point = neighbours[j]
        if labels[point]==-1:
            labels[point]=id
        elif labels[point]==0:
            labels[point] =id
            pointneighbours = region(xlist,point,eps)
            if(len(pointneighbours)>=minpts):
                neighbours = neighbours+pointneighbours
        j=j+1
#         print(j)


# In[24]:


def region(xlist,point,eps):
    neighbours=[]
    for i in range(len(xlist)):
        if np.linalg.norm(xlist[point]-xlist[i])<eps:
            neighbours.append(i)
    return neighbours


# # plotting for dataset1

# In[45]:


xlist = np.genfromtxt("dataset1.txt")

labels=[]
for i in range(len(xlist)):
    labels.append(0)
labels = dBSCAN(xlist,0.2,20)


# print (labels)
# labelset = list(set(labels))
# print (labelset)

plt.title('dbscan clustering for datset1')
col = ['r','b','o']
for i in range(len(xlist)):
    plt.scatter(xlist[i,0],xlist[i,1] , color= col[labels[i]-1] )
plt.show()


# # plotting for datset 2

# In[46]:


xlist = np.genfromtxt("dataset2.txt")
labels=[]
for i in range(len(xlist)):
    labels.append(0)
labels = dBSCAN(xlist,3,10)
# clustering = DBSCAN(eps=0.4, min_samples=10).fit(xlist)
# labels1 = clustering.labels_
# print(len(xlist))
plt.xlabel('k')

plt.title('dbscan clustering for datset2')
# print(labels)
col = ['r','b','y','g','c']
for i in range(len(xlist)):
    plt.scatter(xlist[i,0],xlist[i,1] , color= col[labels[i]+1] )
plt.show()

