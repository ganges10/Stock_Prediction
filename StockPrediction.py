
# coding: utf-8

# In[14]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import sys


# In[15]:

data= pd.read_csv("TITAN.csv")
data.head()


# In[16]:

data['DEX']=np.select([data['Prev Close'] < data['Close'],data['Prev Close'] > data['Close']],[1,-1],1)
print(data.head())


# In[22]:

X_train,X_test=train_test_split(data,test_size=0.2,random_state=42)
print(len(X_train))
print(len(X_test))


# In[ ]:



