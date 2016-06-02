# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from time import time

import metrics
import random
import sys
import numpy as np
import features


def training_label(reference, candidate1, candidate2):
    (_, target1, _) = candidate1
    (_, target2, _) = candidate2

    if metrics.bleu(reference, target1) > metrics.bleu(reference, target2):
        return 1
    return -1


def feature_vector(input, candidate1, candidate2):
    (_, input_features) = input
    (_, _, candidate1_features) = candidate1
    (_, _, candidate2_features) = candidate2

    input_features = np.array(input_features)
    candidate_features = np.array(candidate1_features) - np.array(candidate2_features)

    return np.concatenate((input_features, candidate_features))


def training_label_and_feature_vector(input, reference, candidate1, candidate2, params=(False, False, False, False)):
    input = features.input_features(input, params)
    candidate1 = features.candidate_features(candidate1, params)
    candidate2 = features.candidate_features(candidate2, params)

    return training_label(reference, candidate1, candidate2), feature_vector(input, candidate1, candidate2)


def pro(inputs, references, candidates, sample_size=100, params=(False, False, False, False), seed=None):

    t0 = time()
    random.seed(seed)

    (_, fs_example) = training_label_and_feature_vector(inputs[0], references[0], candidates[0][1], candidates[0][1], params)

    dem1 = len(inputs) * sample_size
    dem2 = len(fs_example)

    x = np.empty((dem1, dem2), dtype=float)
    y = np.empty(dem1, dtype=int)

    k = 0

    for i, inp in enumerate(inputs):

        if i % 100:
            sys.stdout.write("\rPro %6.2f%%" % ((100 * i) / float(len(inputs))))
            sys.stdout.flush()

        candidate = candidates[i]
        reference = references[i]

        for j in range(0, sample_size):

            # Randomly pick two candidates that are not the same
            j1 = j2 = random.randint(0, len(candidate) - 1)
            while j1 == j2:
                j2 = random.randint(0, len(candidate) - 1)

            (ys, fs) = training_label_and_feature_vector(inp, reference, candidate[j1], candidate[j2], params)

            x[k] = fs
            y[k] = ys

            k += 1

    print("\rPro 100.00%")

    pro_time = (time() - t0)
    print("done in %0.3fs" % pro_time)

    return x, y
