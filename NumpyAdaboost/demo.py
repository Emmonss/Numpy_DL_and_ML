import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import cross_val_score


def Makedata():
    x1,y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2,
                                    n_classes=2, shuffle=True, random_state=1)
    x2,y2 = make_gaussian_quantiles(mean=(3,3), cov=1.5, n_samples=300, n_features=2,
                                    n_classes=2, shuffle=True, random_state=1)

    X = np.vstack((x1, x2))
    Y = np.hstack((y1, 1 - y2))
    return X,Y
    # plt.scatter(X[:,0],X[:,1],c=Y)
    # plt.show()

def AdaBosst(X,Y):
    #设定弱分类器
    WeakClassifier = DecisionTreeClassifier(max_depth=1)

    #AdaBoost模型
    clf = AdaBoostClassifier(base_estimator=WeakClassifier,
                             algorithm='SAMME',
                             n_estimators=200,
                             learning_rate=0.9)

    clf.fit(X,Y)

    return clf

def MakeResultGraph():
    X, Y = Makedata()
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1
    x1_, x2_ = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))

    clf = AdaBosst(X,Y)
    y_ = clf.predict(np.c_[x1_.ravel(),x2_.ravel()])
    y_ = y_.reshape(x1_.shape)

    score = cross_val_score(clf,X,Y)
    print(score.mean())

    plt.contourf(x1_, x2_, y_, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()

if __name__ == '__main__':
    MakeResultGraph()