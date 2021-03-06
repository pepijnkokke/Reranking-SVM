# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from time import time

import numpy as np
import data
import pro
import evaluation
import os


OUT_DIR = os.path.abspath(os.path.join('..', 'out'))


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
    # classifier = SVC(kernel='linear', verbose=True)
    classifier = LinearSVC(verbose=True, C=0.0025, tol=1e-8, max_iter=10000)
    # classifier = SGDClassifier(verbose=True, average=100, n_iter=1000)
    classifier.fit(x, y)

    classification_time = (time() - t0)
    print("done in %0.3fs" % classification_time)

    return classifier, classification_time


def test_classifier(classifier, test_x, test_y, train_x, train_y):

    pred_y = classifier.predict(test_x)
    test_score = metrics.accuracy_score(test_y, pred_y)
    print('Score on testing: %0.5f' % test_score)

    pred_y = classifier.predict(train_x)
    train_score = metrics.accuracy_score(train_y, pred_y)
    print('Score on training: %0.5f' % train_score)

    return test_score, train_score


def get_dev_and_processing(n_pca=100, train_input_size=2900, train_sample_size=1000, params=(False, False, False, False)):

    (dev_inputs, dev_references, dev_candidates) = data.load_dev(train_input_size)
    (X_train, y_train) = pro.pro(dev_inputs, dev_references, dev_candidates, train_sample_size, params)

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


def get_test(test_data, pca, normalizer, test_sample_size=2, params=(False, False, False, False)):

    (test_inputs, test_references, test_candidates) = test_data
    (X_test, y_test) = pro.pro(test_inputs, test_references, test_candidates, test_sample_size, params, seed=10)

    print("Normalizing data")
    normalizer.fit(X_test)
    X_test = normalizer.transform(X_test)

    if pca is not None:
        print("Projecting the test data on the orthonormal basis")
        X_test = pca.transform(X_test)

    print("Test set size: %d" % len(X_test))

    return X_test, y_test


def run():

    test_size = 10

    matrix = [
        # name                              # n_pca # corpus size # sample size # pos # ex pos # bigrams # embeddings # combinations
        ('svm-test',                        0, test_size, test_size, False, False, False, False, True),
        ('svm-100-basic',                   0, 100, 200, False, False, False, False, False),
        ('svm-100-basic-500',               0, 100, 500, False, False, False, False, False),
        ('svm-100-basic-1000',              0, 100, 1000, False, False, False, False, False),
        ('svm-100-combinations',            0, 100, 200, True, False, False, False, True),
        ('svm-100-pos',                     0, 100, 200, True, False, False, False, False),
        ('svm-100-pos-500',                 0, 100, 500, True, False, False, False, False),
        ('svm-100-pos-1000',                0, 100, 1000, True, False, False, False, False),
        ('svm-100-ex-pos',                  0, 100, 200, True, True, False, False, False),
        ('svm-100-pos-bigrams',             0, 100, 200, True, False, True, False, False),
        ('svm-100-ex-pos-bigrams',          0, 100, 200, True, True, True, False, False),
        ('svm-100-embedding',               0, 100, 200, False, False, False, True, False),
        ('svm-100-full',                    0, 100, 200, True, True, True, True, True),
        ('svm-500-basic',                   0, 500, 100, False, False, False, False, False),
        ('svm-500-combinations',            0, 500, 100, False, False, False, False, True),
        ('svm-500-pos',                     0, 500, 100, True, False, False, False, False),
        ('svm-500-ex-pos',                  0, 500, 100, True, True, False, False, False),
        ('svm-500-pos-bigrams',             0, 500, 100, True, False, True, False, False),
        ('svm-500-ex-pos-bigrams',          0, 500, 100, True, True, True, False, False),
        ('svm-500-embedding',               0, 500, 100, False, False, False, True, False),
        ('svm-500-full',                    0, 500, 100, True, True, True, True, True),
        ('svm-2900-full',                   0, 2900, 100, True, True, True, True, True),
        ('svm-2900-basic',                  0, 2900, 100, False, False, False, False, False),
        ('svm-2900-basic-200',              0, 2900, 200, False, False, False, False, False),
        ('svm-2900-basic-300',              0, 2900, 300, False, False, False, False, False),
        ('svm-2900-combinations',           0, 2900, 100, True, False, False, False, True),
        ('svm-2900-pos',                    0, 2900, 100, True, False, False, False, False),
        ('svm-2900-ex-pos',                 0, 2900, 100, True, True, False, False, False),
        ('svm-2900-pos-bigrams',            0, 2900, 100, True, False, True, False, False),
        ('svm-2900-ex-pos-bigrams',         0, 2900, 100, True, True, True, False, False),
        ('svm-2900-embedding',              0, 2900, 100, False, False, False, True, False),
        ('svm-2900-full-200',               0, 2900, 200, True, True, True, True, True),
    ]

    # Preload test data into memory
    test_data = data.load_test(2100)

    for name1, n_pca, train_input_size, train_sample_size, pos, extended_pos, bigrams, embeddings, combinations in matrix:

        for i in range(0, 5):

            name = str(i) + '-' + name1

            print("---------------")
            print(name)
            print("---------------")
            t0 = time()

            params = (pos, extended_pos, bigrams, embeddings, combinations)

            (X_train, y_train, normalizer, pca, pca_time) = \
                get_dev_and_processing(n_pca, train_input_size, train_sample_size, params)

            feature_length = len(X_train[0])

            classifier, classification_time = train_classifier(X_train, y_train)

            X_test, y_test = get_test(test_data, pca, normalizer, params=params)
            test_score, train_score = test_classifier(classifier, X_test, y_test, X_train, y_train)

            # clear some memory
            del X_train
            del y_train

            limit = 1000
            if train_input_size == test_size:
                limit = test_size

            blue_baseline, blue, blue_diff, sentences = \
                evaluation.evaluation(test_data, classifier, normalizer, pca, params, limit)

            complete_time = (time() - t0)
            print("Completed in %0.3fs" % complete_time)

            if not os.path.isdir(OUT_DIR):
                os.makedirs(OUT_DIR)

            path = os.path.join(OUT_DIR, 'eval.out')
            with open(path, "a") as eval_file:
                eval_file.write("%-30s , %0.10f , %0.10f , %0.10f , %4d , %8.2f , %8.2f , %8.2f , %4d , %0.10f , %0.10f\n" % (name, blue_baseline, blue, blue_diff, feature_length, classification_time, pca_time, complete_time, train_sample_size, test_score, train_score))

            path = os.path.join(OUT_DIR, '%s.out' % name)
            with open(path, "w") as file:
                file.write('\n'.join(sentences))

if __name__ == "__main__":
    run()
