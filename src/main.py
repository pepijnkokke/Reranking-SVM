# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from time import time

import numpy as np
import data
import pro
import evaluation
import os


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


def train_classifier(x, y):

    t0 = time()
    print('Training classifier')
    classifier = SVC(kernel='linear', C=0.025, verbose=True)
    classifier.fit(x, y)

    classification_time = (time() - t0)
    print("done in %0.3fs" % classification_time)

    return classifier, classification_time


def test_classifier(classifier, test_x, test_y):

    pred_y = classifier.predict(test_x)
    score = metrics.accuracy_score(test_y, pred_y)
    print('Score on testing: %0.5f' % score)

    # print(metrics.classification_report(test_y, pred_y))


def get_dev_and_processing(n_pca=100, train_input_size=2900, train_sample_size=1000, params=(False, False, False, False)):

    (dev_inputs, dev_references, dev_candidates) = data.load_dev(train_input_size, params)
    (X_train, y_train) = pro.pro(dev_inputs, dev_references, dev_candidates, train_sample_size)

    # clear some memory
    del dev_inputs
    del dev_references
    del dev_candidates

    X_train = np.array(X_train)

    print("Normalizing data")
    normalizer = Normalizer(copy=False)
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)

    if n_pca != 0:
        print("PCA for the top %d features from %d features" % (n_pca, len(X_train[0])))
        t0 = time()
        pca = PCA(copy=False, n_components=n_pca)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        pca_time = (time() - t0)
        print("done in %0.3fs" % pca_time)
    else:
        print("No PCA for %d features" % (len(X_train[0])))
        pca = None
        pca_time = 0

    print("Training set size: %d" % len(X_train))

    return X_train, y_train, normalizer, pca, pca_time


def get_test(pca, normalizer, test_input_size=2100, test_sample_size=5, params=(False, False, False, False)):

    (test_inputs, test_references, test_candidates) = \
        data.load_test(test_input_size, params)
    (X_test, y_test) = pro.pro(test_inputs, test_references, test_candidates, test_sample_size)

    print("Normalizing data")
    normalizer.fit(X_test)
    X_test = normalizer.transform(X_test)

    if pca is not None:
        print("Projecting the test data on the orthonormal basis")
        X_test = pca.transform(X_test)

    print("Test set size: %d" % len(X_test))

    return X_test, y_test, test_inputs, test_references, test_candidates


def run():

    matrix = [
        ('svm-100-baseline',                0, 100, 200, 2100, 5, False, False, False, False),
        ('svm-100-full',                    0, 100, 200, 2100, 5, True, True, True, True),
        ('svm-500-baseline',                0, 500, 200, 2100, 5, False, False, False, False),
        ('svm-500-pos',                     0, 500, 200, 2100, 5, True, False, False, False),
        ('svm-500-extended-pos',            0, 500, 200, 2100, 5, True, True, False, False),
        ('svm-500-pos-bigrams',             0, 500, 200, 2100, 5, True, False, True, False),
        ('svm-500-ex-pos-bigrams',          0, 500, 200, 2100, 5, True, True, True, False),
        ('svm-500-representation',          0, 500, 200, 2100, 5, False, False, True, True),
        ('svm-500-full',                    0, 500, 200, 2100, 5, True, True, True, True),
        ('svm-2900-baseline',               0, 2900, 200, 2100, 5, False, False, False, False),
        ('svm-2900-pos',                    0, 2900, 200, 2100, 5, True, False, False, False),
        ('svm-2900-ex-pos',                 0, 2900, 200, 2100, 5, True, True, False, False),
        ('svm-2900-pos-bigrams-pca',        100, 2900, 200, 2100, 5, True, False, True, False),
        ('svm-2900-ex-pos-bigrams-pca',     100, 2900, 200, 2100, 5, True, True, True, False),
        ('svm-2900-representation-pca',     100, 2900, 200, 2100, 5, False, False, True, True),
        ('svm-2900-full-pca',               100, 2900, 200, 2100, 5, True, True, True, True),
        ('svm-2900-pos-bigrams',            0, 2900, 200, 2100, 5, True, False, True, False),
        ('svm-2900-ex-pos-bigrams',         0, 2900, 200, 2100, 5, True, True, True, False),
        ('svm-2900-representation',         0, 2900, 200, 2100, 5, False, False, True, True),
        ('svm-2900-full',                   0, 2900, 200, 2100, 5, True, True, True, True),
    ]

    for name, n_pca, train_input_size, train_sample_size, test_input_size, \
      test_sample_size, pos, extended_pos, bigrams, vector in matrix:

        print("---------------")
        print(name)
        print("---------------")

        params = (pos, extended_pos, bigrams, vector)

        (X_train, y_train, normalizer, pca, pca_time) = \
            get_dev_and_processing(n_pca, train_input_size, train_sample_size, params)

        feature_length = len(X_train[0])

        classifier, classification_time = train_classifier(X_train, y_train)

        # clear some memory
        del X_train
        del y_train

        (X_test, y_test, test_inputs, test_references, test_candidates) = \
            get_test(pca, normalizer, test_input_size, test_sample_size, params)

        test_classifier(classifier, X_test, y_test)

        blue = evaluation.evaluation(test_inputs, test_references, test_candidates, classifier, normalizer, pca)

        path = os.path.join(data.OUT_DIR, 'eval.out')
        with open(path, "a") as eval_file:
            eval_file.write("%-30s , %0.10f , %4d , %8.2f , %8.2f , %4d\n" % (name, blue, feature_length, classification_time, pca_time, train_sample_size))


if __name__ == "__main__":
    run()


