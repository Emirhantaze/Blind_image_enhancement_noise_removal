import numpy as np
from numpy.core.fromnumeric import shape
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import SGDClassifier
import sklearn.model_selection as model_selection
from sklearn import svm
from sklearn.metrics import confusion_matrix
df = pd.read_csv("Variance_info_for_each_channel.csv")
df2 = pd.read_csv("Laplacian_Variance_info.csv")

data = df.to_numpy()
# a = np.concatenate((np.arange(0, 1500, dtype=int),
#                    np.arange(6110, 7700, dtype=int)))

# X = data[a]
X = data[:, [1, 2, 3]]
X[:, 0] = X[:, 0]/X[:, 0].max()
X[:, 1] = X[:, 1]/X[:, 1].max()
X[:, 2] = X[:, 2]/X[:, 2].max()

x4 = np.reshape((X[:, 0]+X[:, 1]+X[:, 2])/3, (X.shape[0], 1))
print(X.shape, x4.shape)
X = np.concatenate(
    (X, x4, np.reshape(df2.iloc[:, 1].to_numpy()/df2.iloc[:, 1].to_numpy().max(), (X.shape[0], 1))), axis=1)
x4 = X[:, 3]*X[:, 4]

X = np.concatenate((X, np.reshape(x4, (X.shape[0], 1))), axis=1)


Y = data[:, 4]


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, Y, train_size=0.6, test_size=0.4, random_state=101)

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=500)
clf = svm.SVC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
matrix = confusion_matrix(y_test, y_pred)
print(matrix.diagonal()/matrix.sum(axis=1))