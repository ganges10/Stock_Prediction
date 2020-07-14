#Numpy is used for mathematical operations
import numpy as np
#Pandas is required to handle with the dataframe
import pandas as pd
#Matplot library is used to obtained the data visualisation of the dataset
import matplotlib.pyplot as plt
#candlestick_ohlc is used to produce the candlestick plot
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
#train_test_split splits the dataset into training and testing data as specified
from sklearn.model_selection import train_test_split
#svm is used to perform support vector classification on the dataset
from sklearn import svm
#Importing confusion_matrix from scikit-learn metrics module for computing confusion matrix
from sklearn.metrics import confusion_matrix 
#Importing accuracy_score from scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score   
#Importing classification_report from scikit-learn metrics module for calculating precision recall and f1 score
from sklearn.metrics import classification_report

def preProcess(data):
    #generating dex column
    data['DEX']=np.select([data['Prev Close'] < data['Close'],data['Prev Close'] > data['Close']],[1,-1],1)
    #considering the important features for X and label values in y
    X=pd.DataFrame(data[['High','Low','Open','Close']])
    y=pd.DataFrame(data.iloc[:,-1])
    #splitting dataset into training and testing data
    return train_test_split(X,y,test_size=0.2,random_state=5)



def histogram(data):
    #histogram for high-low
    df=pd.DataFrame(data[['High','Low']])
    #print(df.head())
    ax = df.plot.hist(bins=12, alpha=0.5)
    plt.show()
    
    #histogram for open close
    df1=pd.DataFrame(data[['Open','Close']])
    #print(df1.head())
    ax = df1.plot.hist(bins=12, alpha=0.5)
    plt.show()
    

def candleStickPlot(data):
    #Considering the Date Open High Low and Close features from the selected datset
    ohlc = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
    ohlc['Date'] = pd.to_datetime(ohlc['Date'])
    ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
    ohlc = ohlc.astype(float)
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    # Setting label & title
    ax.set_xlabel('Date')
    fig.suptitle('Daily Candlestick Chart of NIFTY50')
    #formatting the date into year-month-day format
    date_format = mpl_dates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    fig.tight_layout()   
    plt.show() 
    
    
def timeSeriesPlot(data):
    #plotting the time series for close values over the duration
    data.plot(kind='line',x=data.columns[0],y=data.columns[8])
    plt.xlabel('Date')
    plt.ylabel('Close') 
    plt.show()

def supportVectorClassification(X_train,X_test,Y_train,Y_test):
    #SVC has hyperparamater c gamma and kernel.By default the kernel is Linear
    #C is the normalisation value and is set to 10
    classification=svm.SVC(C=10)
    
    #The training lables are adjusted as 1d-array for training of the model
    y_trainAdjusted=Y_train.values.ravel()
    
    #The classifier is trained based on the obtained training set
    classification.fit(X_train,y_trainAdjusted)
    
    #Prediction is obtained from the trained model for the test dataset.
    y_pred=classification.predict(X_test)
    
    #The accuracy along with the precision recall and f1 score are calculated and displayed
    confusionMatrix(y_pred , Y_test)


def RandomForestAcc(X_train,X_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    #print(clf)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train.values.ravel())
    
    # prediction on test set
    y_pred=clf.predict(X_test)
    
    confusionMatrix(y_pred,y_test)



# Model Accuracy, how often is the classifier correct?
def confusionMatrix(y_pred , y_test):
    #producing the confusion matrix based on the predicted label and actual label
    results = confusion_matrix(y_test,y_pred)
    print("\nConfusion Matrix :")
    print(results) 
    #displaying the precision recall and f1 scores
    print('\nReport : ')
    print(classification_report(y_test, y_pred))
    #Accuracy of the classifier is calculated
    print('\n\nAccuracy Score :',accuracy_score(y_test,y_pred))
    
    

def main():
    #read data from TITAN.csv 
    data=pd.read_csv('TITAN.csv')
    print(data.head())
    print("\n\nThe data is the price history and trading volumes of the TITAN stock in the index NIFTY 50 from NSE (National Stock Exchange) India.")
    print("\n\nFeatures of the dataset:\n",data.columns)

    print("\n\t\tTime Series plot\n")
    timeSeriesPlot(data)
    
    print("\n\t\tCandlestick plot\n")
    candleStickPlot(data)
    
    #data visualization
    print("\n\t\tHistogram plots\n")
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
    X_train,X_test,Y_train,Y_test = preProcess(data)
    print("\n\nAfter splitting the dataset into training and testing data \nThe number of training data is:", len(X_train),"\nThe number of testing data is:",len(X_test))

    print("\n\n\t\tPredicting using Random Forest Classifier\n")
    #random forest and its accuracy
    RandomForestAcc(X_train,X_test,Y_train,Y_test)
    
    print("\n\n\t\tPredicting using Support Vector Classifier\n")
    #using the same training and testing set for Support Vector Classifier
    supportVectorClassification(X_train,X_test,Y_train,Y_test)
    

if __name__=="__main__": 
    main() 
