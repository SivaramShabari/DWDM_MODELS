import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


data = pd.read_csv ("placement_kaggle.csv")

data['status']= data['status'].map({'Placed':1,'Not Placed':0})
data['workex']= data['workex'].map({'Yes':1,'No':0}) 
data['gender']= data['gender'].map({'M':1,'F':0})
data['hsc_b']= data['hsc_b'].map({'Central':1,'Others':0})
data['ssc_b']= data['ssc_b'].map({'Central':1,'Others':0})
data['degree_t']= data['degree_t'].map({'Sci&Tech':0,'Comm&Mgmt':1,'Others':2})
data['specialisation']= data['specialisation'].map({'Mkt&HR':1,'Mkt&Fin':0})
data['hsc_s']= data['hsc_s'].map({'Commerce':0,'Science':1,'Arts':2})




attributes = np.column_stack([data.gender, data.ssc_p, data.ssc_b, data.hsc_p, data.hsc_b, data.hsc_s, data.degree_p,
       data.degree_t, data.workex, data.etest_p, data.specialisation, data.mba_p])
print(attributes)

X_train, X_test, Y_train, Y_test = train_test_split(attributes, data.status, test_size=0.2,random_state=47)

from sklearn.svm import SVC
rbf = SVC(kernel='rbf', gamma = 0.6, random_state = 27)
rbf.fit(X_train,Y_train)

Y_pred = rbf.predict(X_test)


accuracy = np.mean(Y_pred == Y_test)
print("Accuracy for RBF with gamma=0.6: ", accuracy)


from sklearn.metrics import confusion_matrix
cm_rbf = confusion_matrix(Y_test,Y_pred)
print(cm_rbf)


X_train, X_test, Y_train, Y_test = train_test_split(attributes, data.status, test_size=0.2,random_state=98) 
linear = SVC(kernel='linear', random_state = 25)
linear.fit(X_train,Y_train)

Y_pred = linear.predict(X_test)

accuracy = np.mean(Y_pred == Y_test)
print("\nAccuracy for linear: ", accuracy)


from sklearn.metrics import confusion_matrix
cm_rbf = confusion_matrix(Y_test,Y_pred)
print(cm_rbf)


att = np.column_stack([data.workex, data.etest_p])

X_train, X_test, Y_train, Y_test = train_test_split(att, data.status, test_size=0.2,random_state=98) 
cla1 = SVC(kernel='rbf', gamma = 0.6, random_state = 27)
cla1.fit(X_train,Y_train)
Y_pred = cla1.predict(X_test)

X_train, X_test, Y_train, Y_test = train_test_split(att, data.status, test_size=0.2,random_state=98) 
cla2 = SVC(kernel='linear', random_state = 25)
cla2.fit(X_train,Y_train)
Y_pred = cla2.predict(X_test)

import matplotlib.pyplot as plt

h = .02  
X, y = X_train, Y_train


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
titles = ['Kernel Type = RBF, gamma=0.6','Kernel Type = Linear']


for i, clf in enumerate((cla1, cla2)):
    
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Dark2, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Pastel1)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()