# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from sklearn import metrics
from sklearn import svm

import data
import pro
import evaluation

(dev_inputs, dev_references, dev_candidates) = data.load_dev(limit=100)

(train_x, train_y) = pro.pro(dev_inputs, dev_references, dev_candidates, sample_size=10)

classifier = svm.LinearSVC()
classifier.fit(train_x, train_y)

(test_inputs, test_references, test_candidates) = data.load_test(limit=25)

# Small test for the classifier
(test_x, test_y) = pro.pro(test_inputs, test_references, test_candidates, sample_size=10)
pred_y = classifier.predict(test_x)
score = metrics.accuracy_score(test_y, pred_y)
print('Score: ' + str(score))

evaluation.print_evaluation(test_inputs, test_references, test_candidates, classifier)