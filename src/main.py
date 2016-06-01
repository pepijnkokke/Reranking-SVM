# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from time import time

import numpy as np
import data
import pro
import evaluation
import os


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

    path = os.path.join(data.OUT_DIR, 'classifier.%s.pkl' % name)

    if os.path.isfile(path):
        t0 = time()
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


def test_classifier(classifier, test_x, test_y, inputs, candidates, normalizer, pca):

    pred_y = classifier.predict(test_x)
    score = metrics.accuracy_score(test_y, pred_y)
    print('Score on testing: %0.5f' % score)

    # print(metrics.classification_report(test_y, pred_y))

    feature_vector = [pro.feature_vector(inputs[0], candidates[0][0], candidates[0][500])]
    if normalizer is not None:
        feature_vector = normalizer.transform(feature_vector)
    if pca is not None:
        feature_vector = pca.transform(feature_vector)

    prediction1 = classifier.predict(feature_vector)
    print('Prediction 1: %d' % prediction1[0])

    feature_vector = [pro.feature_vector(inputs[0], candidates[0][500], candidates[0][0])]
    if normalizer is not None:
        feature_vector = normalizer.transform(feature_vector)
    if pca is not None:
        feature_vector = pca.transform(feature_vector)

    prediction1 = classifier.predict(feature_vector)
    print('Prediction 2: %d' % prediction1[0])


def get_preprocessed_data(n_components=100, train_input_size=2900, train_sample_size=1000, test_input_size=2100,
                          test_sample_size=5, pos=True, extended_pos=True, bigrams=True, vector=True):

    path_pca = os.path.join(data.OUT_DIR, 'pca-%d-%d-%d-%i%i%i%i.out' %
                            (n_components, train_input_size, train_sample_size, pos, extended_pos, bigrams, vector))
    path_norm = os.path.join(data.OUT_DIR, 'norm-%d-%d-%d-%i%i%i%i.out' %
                            (n_components, train_input_size, train_sample_size, pos, extended_pos, bigrams, vector))

    if os.path.isfile(path_norm):
        normalizer = joblib.load(path_norm)
    else:
        normalizer = Normalizer(copy=False)

    if n_components == 0:
        pca = None
    else:
        if os.path.isfile(path_pca):
            pca = joblib.load(path_pca)
        else:
            pca = PCA(copy=False, n_components=n_components)

    (dev_inputs, dev_references, dev_candidates) = data.load_dev(train_input_size, pos, extended_pos, bigrams, vector)
    (X_train, y_train) = pro.pro(dev_inputs, dev_references, dev_candidates, train_sample_size)

    # clear some memory
    del dev_inputs
    del dev_references
    del dev_candidates

    X_train = np.array(X_train)

    print("Normalizing data")
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)

    if pca is not None:
        print("PCA for the top %d features from %d features" % (n_components, len(X_train[0])))
        t0 = time()
        pca.fit(X_train)
        joblib.dump(pca, path_pca)
        X_train = pca.transform(X_train)
        print("done in %0.3fs" % (time() - t0))

    print("Training set size: %d" % len(X_train))

    (test_inputs, test_references, test_candidates) = data.load_test(test_input_size, pos, extended_pos, bigrams, vector)
    (X_test, y_test) = pro.pro(test_inputs, test_references, test_candidates, test_sample_size)

    print("Normalizing data")
    normalizer.fit(X_test)
    X_test = normalizer.transform(X_test)

    if pca is not None:
        print("Projecting the test data on the orthonormal basis")
        X_test = pca.transform(X_test)

    print("Test set size: %d" % len(X_test))

    return X_train, y_train, X_test, y_test, normalizer, pca, test_inputs, test_references, test_candidates


def run():

    matrix = [
        ('svm-100-baseline', 0, 100, 100, 100, 5, False, False, False, False,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-500-baseline', 0, 500, 100, 200, 5, False, False, False, False,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-2700-baseline', 0, 2700, 100, 2100, 5, False, False, False, False,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-2700-pos', 0, 2700, 100, 2100, 5, True, False, False, False,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-2700-extended-pos', 0, 2700, 100, 2100, 5, True, True, False, False,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-2700-pos-bigrams', 0, 2700, 100, 2100, 5, True, False, True, False,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-2700-extended-pos-bigrams', 0, 2700, 100, 2100, 5, True, True, True, False,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-2700-representation', 0, 2700, 100, 2100, 5, False, False, True, True,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
        ('svm-2700-full', 0, 2700, 100, 2100, 5, True, True, True, True,
         lambda: SVC(kernel='linear', C=0.025, verbose=True)),
    ]

    for name, n_components, train_input_size, train_sample_size, test_input_size, \
      test_sample_size, pos, extended_pos, bigrams, vector, classifier in matrix:

        print("---------------")
        print(name)
        print("---------------")

        (X_train, y_train, X_test, y_test, normalizer, pca, test_inputs, test_references, test_candidates) = \
            get_preprocessed_data(n_components, train_input_size, train_sample_size, test_input_size, \
                test_sample_size, pos, extended_pos, bigrams, vector)

        classifier = train_classifier(classifier, X_train, y_train, name)
        test_classifier(classifier, X_test, y_test, test_inputs, test_candidates, normalizer, pca)
        evaluation.print_evaluation(test_inputs, test_references, test_candidates, classifier, normalizer, pca)


if __name__ == "__main__":
    run()


