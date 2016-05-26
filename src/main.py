# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


import data
import pro
import evaluation

_dev_data = 0
_test_data = 0


def dev_data():
    global _dev_data

    if _dev_data == 0:
        _dev_data = data.load_dev(500)

    return _dev_data


def test_data():
    global _test_data

    if _test_data == 0:
        _test_data = data.load_test(500)

    return _test_data


def train_classifier(classifier):

    (dev_inputs, dev_references, dev_candidates) = dev_data()
    (train_x, train_y) = pro.pro(dev_inputs, dev_references, dev_candidates, sample_size=100)

    print("Training classifier")
    classifier.fit(train_x, train_y)


def evaluate_classifier(classifier):

    (test_inputs, test_references, test_candidates) = test_data()

    (test_x, test_y) = pro.pro(test_inputs, test_references, test_candidates, sample_size=10)
    pred_y = classifier.predict(test_x)
    score = metrics.accuracy_score(test_y, pred_y)
    print('Score: ' + str(score))

    evaluation.print_evaluation(test_inputs, test_references, test_candidates, classifier)


if __name__ == "__main__":

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()
    ]

    for name, classifier in zip(names, classifiers):
        print("---------------")
        print(name)
        train_classifier(classifier)
        evaluate_classifier(classifier)

