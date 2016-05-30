# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from time import time

import numpy as np
import data
import pro
import evaluation
import os
import msgpack
from sklearn.externals import joblib


def dump_to_svm_format(x, y, filename='out.svm'):
    """
    Dump to SVM file to be used in SVM libraries like LaSVM of LIBSVM
    """
    dump = os.path.join(data.OUT_DIR, filename)
    with open(dump, 'w') as stream:
        for i in range(0, len(y)):
            if y[i] == 1:
                stream.write("+1")
            else:
                stream.write("-1")
            for j, feature in enumerate(x[i]):
                stream.write(" " + str(j) + ":" + str(feature))
            stream.write("\n")


def train_classifier(get_classifier, x, y, name='classifier'):
    print("---------------")

    path = os.path.join(data.OUT_DIR, 'classifier.%s.pkl' % name)

    if os.path.isfile(path):
        print('Loading classifier: %s' % name)
        classifier = joblib.load(path)
    else:
        t0 = time()
        print('Training classifier: %s' % name)
        classifier = get_classifier()
        classifier.fit(x, y)
        print("done in %0.3fs" % (time() - t0))
        print('Dumping classifier')
        joblib.dump(classifier, path)

    return classifier


def test_classifier(classifier, x, test_y):

    print('Testing classifier')
    pred_y = classifier.predict(x)
    score = metrics.accuracy_score(test_y, pred_y)
    print('Score: %0.5f' % score)


def get_preprocessed_data(n_components=100, train_input_size=2000, train_sample_size=100, test_input_size=2000, test_sample_size=5):

    path_train = os.path.join(data.OUT_DIR, 'preprocessed-train-%d-%d-%d.out' % (n_components, train_input_size, train_sample_size))
    path_test = os.path.join(data.OUT_DIR, 'preprocessed-test-%d-%d-%d.out' % (n_components, test_input_size, test_sample_size))
    path_pca = os.path.join(data.OUT_DIR, 'pca-%d-%d-%d.out' % (n_components, train_input_size, train_sample_size))

    if os.path.isfile(path_pca):
        pca = joblib.load(path_pca)
    else:
        pca = PCA(copy=True, n_components=n_components)

    if os.path.isfile(path_train):
        with open(path_train, 'r') as stream:
            print('Reading processed dev data')
            (X_train_pca, y_train) = msgpack.unpack(stream, use_list=True)
            X_train_pca = np.array(X_train_pca)
    else:
        (dev_inputs, dev_references, dev_candidates) = data.load_dev(train_input_size)
        (X_train, y_train) = pro.pro(dev_inputs, dev_references, dev_candidates, sample_size=train_sample_size)
        print("Training set size: %d" % len(X_train))

        X_train = np.array(X_train)

        print("Extracting the top %d features from %d features" % (n_components, len(X_train[0])))
        t0 = time()
        pca.fit(X_train)
        joblib.dump(pca, path_pca)
        print("done in %0.3fs" % (time() - t0))

        print("Projecting the training data on the orthonormal basis")
        t0 = time()
        X_train_pca = pca.transform(X_train)
        print("done in %0.3fs" % (time() - t0))

        with open(path_train, 'w') as stream:
            print('Dumping processed dev data')
            dump = (X_train_pca.tolist(), y_train)
            msgpack.pack(dump, stream)

    print("Training set size: %d" % len(X_train_pca))

    if os.path.isfile(path_test):
        with open(path_test, 'r') as stream:
            print('Reading processed test data')
            (X_test_pca, y_test, test_inputs, test_references, test_candidates) = msgpack.unpack(stream, use_list=True)
            X_test_pca = np.array(X_test_pca)
    else:

        (test_inputs, test_references, test_candidates) = data.load_test(test_input_size)
        (X_test, y_test) = pro.pro(test_inputs, test_references, test_candidates, sample_size=test_sample_size)

        print("Projecting the test data on the orthonormal basis")
        t0 = time()
        X_test_pca = pca.transform(X_test)
        print("done in %0.3fs" % (time() - t0))

        with open(path_test, 'w') as stream:
            print('Dumping processed test data')
            dump = (X_test_pca.tolist(), y_test, test_inputs, test_references, test_candidates)
            msgpack.pack(dump, stream)

    print("Test set size: %d" % len(X_test_pca))

    return X_train_pca, y_train, X_test_pca, y_test, pca, test_inputs, test_references, test_candidates


def run():
    (X_train, y_train, X_test, y_test, pca, test_inputs, test_references, test_candidates) = get_preprocessed_data()

    names = [
        "nn1",
        # "nn2",
        # "nn3",
        # "svm1"
        # "svm2"
    ]

    classifiers = [
        lambda: MLPClassifier(hidden_layer_sizes=(10,), activation='tanh', algorithm='sgd', batch_size='auto',
                      learning_rate='adaptive', learning_rate_init=0.01, verbose=True, tol=0.00001, max_iter=1000),
        # lambda: MLPClassifier(hidden_layer_sizes=(500,), activation='tanh', algorithm='sgd', batch_size='auto',
        #               learning_rate='adaptive', learning_rate_init=0.01, verbose=True, tol=0.00001, max_iter=1000),
        # lambda: MLPClassifier(hidden_layer_sizes=(250,150), activation='tanh', algorithm='sgd', batch_size='auto',
        #               learning_rate='adaptive', learning_rate_init=0.01, verbose=True, tol=0.00001, max_iter=1000),
        # lambda: LinearSVC(C=0.025, verbose=True, max_iter=1000),
        # lambda: SVC(kernel='linear', C=0.025, verbose=True)
    ]

    for classifier, name in zip(classifiers, names):
        classifier = train_classifier(classifier, X_train, y_train, name)
        test_classifier(classifier, X_test, y_test)
        evaluation.print_evaluation(test_inputs, test_references, test_candidates, classifier, pca, limit=25)


if __name__ == "__main__":
    run()


