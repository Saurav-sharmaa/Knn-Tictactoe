import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import  neighbors,metrics
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("tictac.data")
# print(data.head())

X = data[[
    'top-left-square',
    'top-middle-square',
    'top-right-square',
    'middle-left-square',
    'middle-middle-square',
    'middle-right-square',
    'bottom-left-square',
    'bottom-middle-square',
    'bottom-right-square'
]].values

y = data[['class']]
#tranforming x
le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = le.fit_transform(X[:,i])
# print(X)

#transforming y

label_maping = {
    'positive': 1,
    'negative':0
}

y['class'] = y['class'].map(label_maping)
a
y = np.array(y)
# print(y)

#test and train
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.25)

#model
knn = neighbors.KNeighborsClassifier(n_neighbors=21,weights="uniform")

knn.fit(X_train,y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test,prediction)

print(f"Prediction: {prediction}\naAccuracy: {accuracy}")






