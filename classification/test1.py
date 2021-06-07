from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import SGDClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# TODO add scaler and delete all the other scaling operators here
# https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/

df = pd.read_csv("Variance_info.csv")
df2 = pd.read_csv("Laplacian_Variance_info.csv")
df3 = pd.read_csv("One_Zero_info.csv")
df4 = pd.read_csv("FFT_info.csv")
df4 = df4.fillna(0.5)
data = df.to_numpy()
# a = np.concatenate((np.arange(0, 1500, dtype=int),
#                    np.arange(6110, 7700, dtype=int)))

# X = data[a]
X = data[:, [1, 2, 3]]
X[:, 0] = X[:, 0]/X[:, 0].max()
X[:, 1] = X[:, 1]/X[:, 1].max()
X[:, 2] = X[:, 2]/X[:, 2].max()

x4 = np.reshape((X[:, 0]+X[:, 1]+X[:, 2])/3, (X.shape[0], 1))
X = np.concatenate(
    (X, x4, np.reshape(df2.iloc[:, 1].to_numpy()/df2.iloc[:, 1].to_numpy().max(), (X.shape[0], 1))), axis=1)
x4 = X[:, 3]*X[:, 4]

X = np.concatenate((X, np.reshape(x4, (X.shape[0], 1))), axis=1)
x1 = df3.iloc[:, 1]
x2 = df3.iloc[:, 2]
x1 = x1 / x1.max()
x1 = np.reshape(x1.to_numpy(), (X.shape[0], 1))
x2 = x2 / x2.max()
x2 = np.reshape(x2.to_numpy(), (X.shape[0], 1))
X = np.concatenate((X, x1, x2), axis=1)
i = list(range(9))
i.pop(4)
i = np.array(i)
X = np.concatenate((X, df4.to_numpy()[:, i+1]), axis=1)
X = np.concatenate((X, df4.to_numpy()[:, i+10]), axis=1)
X = np.concatenate((X, df4.to_numpy()[:, i+19]), axis=1)
X = np.concatenate((X, df4.to_numpy()[:, [28, 29, 30]]), axis=1)


Y = data[:, 4]


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, Y, train_size=0.6, test_size=0.4, random_state=101)

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=500)
clf = svm.SVC()
clf = NearestCentroid()
clf = KNeighborsClassifier()
clf = GaussianProcessClassifier()
clf3 = GaussianNB()
clf1 = tree.DecisionTreeClassifier()  # 2nd best working for now
clf = RandomForestClassifier(
    criterion="gini",    n_estimators=100, random_state=101)  # best working for now


clf2 = LogisticRegression(random_state=1)
clf3 = MLPClassifier(hidden_layer_sizes=(
    15,), random_state=1, max_iter=1000, warm_start=True)
clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf), ('gnb', clf3)],
    voting='hard')


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
matrix = confusion_matrix(y_test, y_pred)
print(matrix/matrix.sum(axis=1))
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
