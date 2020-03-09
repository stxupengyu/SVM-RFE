import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train[['HMM', 'SSD', 'OGS']].values
y_train = train[['Target']].values.ravel()
X_test = test[['HMM', 'SSD', 'OGS']].values
y_test = test[['Target']].values.ravel()

#(a)
estimator = svm.SVC(kernel="linear")
estimator.fit(X_train, y_train)
a_result = estimator.predict(X_test)
print("accuracy: ", accuracy_score(y_test,a_result))
print("weights of HMM, SSD, OGS: ", estimator.coef_)

#(b)1
selector = RFE(estimator, 2, step=1)
selector = selector.fit(X_train, y_train)
#record the selected variable
summary = np.zeros(sum(selector.support_)).tolist()
j=0
k=0
for i in selector.support_:
    j=j+1
    if i==True:
        summary[k]=j-1
        k=k+1
#new X based on selected variable
X_train1 = X_train[:,summary]
X_test1 = X_test[:,summary]
#new fit
estimator.fit(X_train1, y_train)
a_result = estimator.predict(X_test1)

print("accuracy: ", accuracy_score(y_test,a_result))
print("seleted variable",selector.support_)
print("weights of HMM, SSD, OGS: ", estimator.coef_)




#(b)2
selector = RFE(estimator, 1, step=1)
selector = selector.fit(X_train, y_train)
#record the selected variable
summary = np.zeros(sum(selector.support_)).tolist()
j=0
k=0
for i in selector.support_:
    j=j+1
    if i==True:
        summary[k]=j-1
        k=k+1
#new X based on selected variable
X_train2 = X_train[:,summary]
X_test2 = X_test[:,summary]
#new fit
estimator.fit(X_train2, y_train)
a_result = estimator.predict(X_test2)

print("accuracy: ", accuracy_score(y_test,a_result))
print("seleted variable",selector.support_)
print("weights of HMM, SSD, OGS: ", estimator.coef_)

plt.scatter(X_test2,np.zeros(len(X_test2)), marker='o',c=y_test)           
plt.scatter(X_test1[:,0], X_test1[:,1], marker='o',c=y_test)

