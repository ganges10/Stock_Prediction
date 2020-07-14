
# coding: utf-8

# In[14]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import sys


def histogram(data):
    #histogram for high-low
    df=pd.DataFrame(data[['High','Low']])
    print(df.head())
    ax = df.plot.hist(bins=12, alpha=0.5)
    
    #histogram for open close
    df1=pd.DataFrame(data[['Open','Close']])
    print(df1.head())
    ax = df1.plot.hist(bins=12, alpha=0.5)



def RandomForestAcc(X_train,X_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    #print(clf)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    # prediction on test set
    y_pred=clf.predict(X_test)
    
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

def main():
    #read data from TITAN.csv 
    data=pd.read_csv('TITAN.csv')
    print(data.head())
    print(data.columns)

    #data visualization
    histogram(data)
    
    #The above two figures are histograms plotted between
    #CLOSE and OPEN and the attributes HIGH and
    #LOW. This is done because we believe today's closing
    #price and opening price along with the high and lowest price
    #of the stock during last year will affect the price of the stock
    #at a later date. Based on such reasoning we devised a logic
    #if today's CLOSE is greater than yesterday's CLOSE
    #then we assign the value 1 to DEX or else we assign the
    #value -1 to DEX
    
    #generate dex column
    data['DEX']=np.select([data['Prev Close'] < data['Close'],data['Prev Close'] > data['Close']],[1,-1],1)
    print(data.head())

    #consider the following attributes for X and y
    X=pd.DataFrame(data[['High','Low','Open','Close']])
    y=pd.DataFrame(data.iloc[:,-1])
    
    #split training and testing data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)
    print(X_train)
    print(len(X_test))
    print(y_train)
    print(len(y_test))

    #random forest and its accuracy
    RandomForestAcc(X_train,X_test,y_train,y_test)
    

if __name__=="__main__": 
    main() 
