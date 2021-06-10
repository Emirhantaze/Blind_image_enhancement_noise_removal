from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import metrics, svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler as SS
import sklearn.model_selection as model_selection

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pickle import dump
n = [1984, 2008, 2027, 1981]


df = pd.read_csv("Variance_info.csv")
df2 = pd.read_csv("Laplacian_Variance_info.csv")
df3 = pd.read_csv("One_Zero_info.csv")
df4 = pd.read_csv("FFT_info.csv")
df4 = df4.fillna(0.5)
df = df.to_numpy()
df2 = df2.to_numpy()
df3 = df3.to_numpy()
df4 = df4.to_numpy()
n1 = 0
n2 = n1 + sum(n)
Y = np.array(df[n1:n2, 4])
df = np.array(df[n1:n2, [1, 2, 3]], dtype=np.double)
df2 = np.reshape(df2[n1:n2, 1], (n2-n1, 1))
df3 = np.array(df3[n1:n2, [1, 2]], dtype=np.double)
df4 = np.array(df4[n1:n2, 1:31], dtype=np.double)
X = np.concatenate([df, df2, df3, df4], axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, Y, train_size=0.6, test_size=0.4, random_state=101)


scalerX = SS()


scalerX = scalerX.fit(X_train)

print(X[0])

X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)
print(scalerX.transform([X[0]]))

# for i in range(36):
#     plt.scatter(X[:, i], Y, marker="o")
#     plt.title(i)
#     plt.show()

# models = {"SGDClassifier": SGDClassifier(loss="hinge", penalty="l2", max_iter=500), "SVC": svm.SVC(), "NearestCentroid": NearestCentroid(), "KNeighborsClassifier": KNeighborsClassifier(), "GaussianNB": GaussianNB(), "DecisionTreeClassifier": tree.DecisionTreeClassifier(), "RandomForestClassifier": RandomForestClassifier(
#     criterion="gini",    n_estimators=100, random_state=101), "LogisticRegression": LogisticRegression(random_state=1), "MLPClassifier": MLPClassifier(hidden_layer_sizes=(
#         15,), random_state=1, max_iter=1000, warm_start=True), }

# pred = []
# for name, algo in models.items():
#     model = algo
#     model.fit(X_train, y_train.flatten())
#     predictions = model.predict(X_test)
#     acc = metrics.accuracy_score(y_test, predictions)
#     pred.append(acc)
#     print(name, acc)


# clf = RandomForestClassifier(
#     criterion="gini",    n_estimators=100, random_state=101)  # best working for now
# clf1 = tree.DecisionTreeClassifier()
clf3 = MLPClassifier(hidden_layer_sizes=(
    15,), random_state=1, max_iter=1000, warm_start=True)
# clf = VotingClassifier(
#     estimators=[('lr', clf1), ('rf', clf), ('gnb', clf3)],
#     voting='hard')


model = clf3
model.fit(X_train, y_train.flatten())
predictions = model.predict(X_test)
acc = metrics.accuracy_score(y_test, predictions)

print("Voting Classifier", acc)


# sns.barplot(y=list(models.keys()), x=pred,
#             linewidth=1.5, orient='h', edgecolor="0.1")
# plt.xlabel("Accuracy")
# plt.title("Comparison of Different ML models")
# plt.show()
print("Accuracy:", metrics.accuracy_score(y_test, predictions))
matrix = metrics.confusion_matrix(y_test, predictions)
print(matrix/matrix.sum(axis=1))
# metrics.plot_confusion_matrix(clf3, X_test, y_test)
# plt.show()
dump(model, open('models/modelimagetypeest.pkl', 'wb'))
# save the scaler
dump(scalerX, open('models/scalerimagetypeest.pkl', 'wb'))
