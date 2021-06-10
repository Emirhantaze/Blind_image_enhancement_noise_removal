from datetime import timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LR1
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.svm import SVR
from sklearn.linear_model import Ridge as RR

from sklearn.metrics import r2_score, mean_squared_error
import sklearn.model_selection as model_selection

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler as SS

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
n1 = sum(n[:2])
n2 = n1 + n[2]
print(n1, n2)
Y = np.array(df[n1:n2, 8])
df = np.array(df[n1:n2, [1, 2, 3]], dtype=np.double)
df2 = np.reshape(df2[n1:n2, 1], (n2-n1, 1))
df3 = np.array(df3[n1:n2, [1, 2]], dtype=np.double)
df4 = np.array(df4[n1:n2, 1:28], dtype=np.double)
X = np.concatenate([df, df2, df3, df4], axis=1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, Y, train_size=0.6, test_size=0.4, random_state=101)


scalerX = SS()


scalerX = scalerX.fit(X_train)


X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)


# for i in range(36):
#     plt.scatter(X[:, i], Y, marker="o")
#     plt.title(i)
#     plt.show()

models = {'Linear Regression': LR(), 'Decision Tree Regression': DTR(), 'Random Forest Regression': RFR(), 'Gradient Boosting Regression': GBR(
), 'Ada Boosting Regression': ABR(), 'K-Neighbors Regression': KNR(), 'Support Vector Regression': SVR(), 'Ridge Regression': RR(), "MLPRegressor": MLPRegressor(random_state=1, max_iter=5000)}
pred = []
print(models.keys())


for name, algo in models.items():
    model = algo
    model.fit(X_train, y_train.flatten())
    predictions = model.predict(X_test)
    acc = r2_score(y_test, predictions)
    pred.append(acc)
    print(name, acc, " SQRTE: ",    mean_squared_error(y_test, predictions))


# # ('lr',  RFR()), ('rf', GBR()),
# final = RFR(n_estimators=200)
# final = MLPRegressor(random_state=1, max_iter=5000)
# final.fit(X_train, y_train.flatten())
# final_pred = final.predict(X_test)
# acc = r2_score(y_test, final_pred)
# print(acc, " SQRTE: ",    mean_squared_error(y_test, final_pred))
# # plt.scatter(y_test, final_pred)
# # plt.show()
# dump(final, open('models/modelangleest.pkl', 'wb'))
# # save the scaler
# dump(scalerX, open('models/scalerangleest.pkl', 'wb'))
