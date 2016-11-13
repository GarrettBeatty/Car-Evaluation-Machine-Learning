import pandas as pd 
import numpy as np 
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('car.data.txt')
df = df.apply(preprocessing.LabelEncoder().fit_transform)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)