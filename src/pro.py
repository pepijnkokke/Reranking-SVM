# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import metrics
import random
import sys


def training_label(reference, candidate1, candidate2):
    (_, target1, _) = candidate1
    (_, target2, _) = candidate2

    if metrics.bleu(reference, target1) > metrics.bleu(reference, target2):
        return 1

    return 0


def feature_vector(input, candidate1, candidate2):
    (_, input_features) = input
    (_, _, candidate1_features) = candidate1
    (_, _, candidate2_features) = candidate2

    return input_features + candidate1_features + candidate2_features


def pro(inputs, references, candidates, sample_size=100):
    x = []
    y = []

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

            x.append(feature_vector(inp, candidate[j1], candidate[j2]))
            y.append(training_label(reference, candidate[j1], candidate[j2]))

    print("\rPro 100.00%%")

    return x, y
