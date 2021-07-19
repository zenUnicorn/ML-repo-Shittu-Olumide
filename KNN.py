import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#archive.ics.uci.edu/ml/datasets.html

df = pd.read_csv('breast-cancer-wisconsin.data.txt ')
#data contains null values with question mark, so replace ? with -99999
df.replace('?', -99999, inplace=True)
#droping column id cos its an outliers meaning it has no benefit to our dataset.
df.drop(['id'], 1, inplace=True)

#define X=features and y=labels
X = np.array(df.drop(['class'],1)) #this means all the columns are features except the class column so drop it
y = np.array(df['class'])  #this means the label is only the class column

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#define your classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

#to predict with our KNN
example_measures = np.array = np.array([4,2,1,1,1,2,3,2,1])
#getting rid of DeprecationWarning
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)























