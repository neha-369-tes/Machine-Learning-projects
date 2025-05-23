import pandas as pd

dataset = pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator =  classifier, X= X_train, y = y_train, cv=10)
print("accuracy: {:.2f}% ".format(acc.mean()*200))
print("standard deviation: {:.2f}% ".format(acc.std()*200))
