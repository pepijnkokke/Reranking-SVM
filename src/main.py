# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

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


if __name__ == "__main__":

    (dev_inputs, dev_references, dev_candidates) = data.load_dev()
    (train_x, train_y) = pro.pro(dev_inputs, dev_references, dev_candidates, sample_size=200)
    print("Training examples: %d" % len(train_x))
    print("Features : %d" % len(train_x[0]))

    (test_inputs, test_references, test_candidates) = data.load_test(250)
    (test_x, test_y) = pro.pro(test_inputs, test_references, test_candidates, sample_size=2)
    print("Test examples: %d" % len(test_x))

    names = [
        "nn",
        "linearsvm"
    ]

    classifiers = [
        # MLPClassifier(hidden_layer_sizes=(300, 750), activation='relu', algorithm='sgd', batch_size='auto',
        #               learning_rate='adaptive', learning_rate_init=0.01, verbose=True),
        LinearSVC(C=0.025, verbose=True, max_iter=10000)
    ]

    for name, classifier in zip(names, classifiers):
        print("---------------")
        print(name)
        print("Training classifier")
        classifier.fit(train_x, train_y)

        print("Testing classifier")
        pred_y = classifier.predict(test_x)
        score = metrics.accuracy_score(test_y, pred_y)
        print('Score: ' + str(score))

        print("Storing classifier")
        path = os.path.join(data.OUT_DIR, 'classifier.' + name + '.pkl')
        joblib.dump(classifier, path)

        evaluation.print_evaluation(test_inputs, test_references, test_candidates, classifier, top=100)

