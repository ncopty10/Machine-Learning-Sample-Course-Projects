import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def plot_decision_boundary(X, Y, X_label, Y_label):

    model = linear_model.LogisticRegression(C=1e5)
    # Create an instance of Logistic Regression Classifier and fit the data.
    model.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(10, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.title("Decision Boundary")
    plt.show()

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2, label="SVM")
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

if __name__ == "__main__":
    iris = datasets.load_iris()

    X = iris.data[0:100,0:2]  # we only take the first 100 data points and the first 2 features.
    y = iris.target[0:100]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.8, random_state=0) #split data

    #regularize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    # linear model

    lm = LogisticRegression()
    lm.fit(X_train, y_train)

    plot_decision_boundary(X_train,y_train,"Sepal Length", "Width")

    # test accuracy
    lm.predict_proba(X_test)
    print('Model Accuracy= ', lm.score(X_test, y_test))

    lm.predict_proba(X_train)
    print('Model Accuracy= ', lm.score(X_train, y_train))

    #Support Vector Machine

    clf = svm.SVC(kernel='linear', C=float("inf")).fit(X_train, y_train)

    plt.clf()
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    plt.figure(1)
    plot_svc_decision_boundary(clf, -3, 3)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X_train[y_train != 0, 0], X_train[y_train != 0, 1],
                color='blue', marker='x', label='not setosa')
    plt.axis([-2, 2.5, -3, 3.5])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend(loc='upper left')
    plt.show()

    #test accuracy
    print('Model Accuracy= ', clf.score(X_train, y_train))

    print('Model Accuracy= ', clf.score(X_test, y_test))

    dual = clf.dual_coef_

    # Calculate margin
    w = clf.coef_[0]
    b = clf.intercept_[0]

    support_v = clf.support_vectors_
    sup = clf.support_

    sv = support_v[0]
    sv2 = support_v[1]

    dist = (np.dot(w,sv) + b)/np.linalg.norm(w)
    dist2 = (np.dot(w,sv2) + b)/np.linalg.norm(w)

    print("Margin calculated using support vector sv1: ",abs(dist*2))
    print("Margin calculated using support vector sv1: ",abs(dist2*2))

    # Margin is calculated as two times the distance between a support vector and the decision boundary. Two different support vectors are used to verify results.

    # vector orthogonal tp decision boundary
    w = clf.coef_[0]     # = [ 1.93848301, -1.38664541]



    #binary linear classifier and SVM have different decision boundaries since they have different w's and b' as shown below

    w = clf.coef_[0]     # = [ 1.93848301, -1.38664541]
    b = clf.intercept_[0]    # =0.2669280505962728

    w_lm = lm.coef_[0] # = [1.61445904, -1.19407207]
    b_lm = lm.intercept_[0] # = 0.1382520483146005

    #2.8

    #seperate data
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X,y, test_size=0.4, random_state=0)

    #regularize data
    scaler = StandardScaler().fit(X_train2)
    X_train2 = scaler.transform(X_train2)

    X_test2 = scaler.transform(X_test2)

    clf2 = svm.SVC(kernel='linear', C=float("inf")).fit(X_train2, y_train2)

    w2 = clf2.coef_[0]     # = [ 3.82356753, -2.43160619]
    b2 = clf2.intercept_[0]    # = 0.38584432034884864

    # the decision boundary has changed since w2 != w and b2 != b. When the train set changed the support vectors (the data points closest to the decision boundary) have also changed. Therefore the decision boundary has to change to optimize this new train set.

    print('Model Accuracy= ', clf.score(X_train2, y_train2))

    print('Model Accuracy= ', clf.score(X_test2, y_test2))

    # The accuracy did not change since the accuracy is still 100%.


    #plot
    plot_svc_decision_boundary(clf2, -3, 3)
    plt.scatter(X_train2[y_train2 == 0, 0], X_train2[y_train2 == 0, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X_train2[y_train2 != 0, 0], X_train2[y_train2 != 0, 1],
                color='blue', marker='x', label='not setosa')
    plt.axis([-2, 2.5, -3, 3.5])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend(loc='upper left')
    plt.show()

    #2.10

    # To deal with data that is not linearly seperable, we can use polynomial SVM and use soft margin. To do this the kernel is set to 'poly' and the parameter C is set to 1. Previosly, C was set to infinity to prevent misclassifications. When C is set to a lower value, penalty for misclassifications are decreased.

    iris = datasets.load_iris()

    X = iris.data[:,0:2]  # we only take the first two features.
    y = iris.target

    # seperate data
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X,y, test_size=0.01, random_state=0)

    scaler = StandardScaler().fit(X_train3)
    X_train3 = scaler.transform(X_train3)

    X_test3 = scaler.transform(X_test3)

    # train model
    clf3 = svm.SVC(kernel='poly',degree=3,C=1,decision_function_shape='ovo').fit(X_train3,y_train3)


    # plot
    plt.clf()

    h = .01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))


    Z = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("Decision Boundary")
    plt.show()


















