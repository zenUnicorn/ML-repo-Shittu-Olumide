import pandas as pd
import Quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.limear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

####Regression in ML using Stock Market data (Continuous data) ###

df = Quandl.get('WIKI/GOOGL')

#print(df.head())
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

#High and Low margin
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

#Daily percent change or daily move
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#Now lets pick out our needed columns
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#To be able to predict, probably the close price by the end of the day create a variable
forecast_col = 'Adj. Close'
#fill all empty columns
df.fillna(-99999, inplace=True)

#to get the 10% of the dataframe. Or using data that came 10days ago to predict today

forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)

#now to create our labels
#this shifts the forecast_col up to the negative area such that it predicts 10days ago
df['label'] = df[forecast_col].shift[-forecast_out]
df.dropma(inplace=True)

#Feature is capital X
X = no.array(df.drop(['label'],1))

#Now to Scale X
X = preprocessing.scale(X)
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
#Label is lowercase y
y = np.array(df['label'])

#Creating our train and test data.... 0.2 = 20%
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Create your classifier
clf = LinearRegression(n_jobs=10)  #You can used SupportVectorRegression too clf=svm.SVR(kernel='poly')
#fit your train data
clf.fit(X_train, y_train)
#to save your classifier with pickle
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)


#to get your accuracy
accuracy = clf.score(X_test,y_test)
#print(accuracy)

#to predict
forecast_set = clf.predict(X_lately)
#it prints the next set of predicted data, accuracy and number of days
print(forecast_set, accuracy, forecast_out)

#to visualize the data and the predicted data
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.coulmns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()






















