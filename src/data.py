# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import os
import sys
import re

DATA_DIR = os.path.abspath(os.path.join('..', 'data'))
BASELINE_WEIGHTS = os.path.join(DATA_DIR, 'baseline.weights')
DEV_BEST = os.path.join(DATA_DIR, 'nlp2-dev.1000best')
DEV_DE = os.path.join(DATA_DIR, 'nlp2-dev.de')
DEV_EN_PLF = os.path.join(DATA_DIR, 'nlp2-dev.en.pw.plf-100')
DEV_EN = os.path.join(DATA_DIR, 'nlp2-dev.en.s')
TEST_BEST = os.path.join(DATA_DIR, 'nlp2-test.1000best')
TEST_DE = os.path.join(DATA_DIR, 'nlp2-test.de')
TEST_EN_PLF = os.path.join(DATA_DIR, 'nlp2-test.en.pw.plf-100')
TEST_EN = os.path.join(DATA_DIR, 'nlp2-test.en.s')


def read(input_file, reference_file, candidates_file, limit):
    """
    Parameters
    ----------
    limit Limit the number of inputs parsed
    pos Whether or not to include pos features in the feature set
    extended_pos Whether or not to include extended pos features in the feature set
    bigrams Whether or not to include pos bigrams in the feature set
    vector Whether or not to include a vector representation in the feature set

    Returns
    -------
    (inputs, references, candidates)
    """

    with open(input_file, 'r') as f:
        inputs = []
        for i in range(0, limit):
            if i % 100:
                sys.stdout.write("\rInputs %6.2f%%" % ((100 * i) / float(limit)))
                sys.stdout.flush()
            inputs.append(f.readline())
        print("\rInputs 100.00%")

    with open(reference_file, 'r') as f:
        references = [f.readline().strip(' \t\n\r') for i in range(0, limit)]

    with open(candidates_file, 'r') as f:
        candidates = []
        candidate_set = []
        i = 0

        while True:
            candidate = parse_candidate(f.readline())
            j = candidate[0]

            if j == i:
                candidate_set.append(candidate)
            else:
                candidates.append(candidate_set)
                candidate_set = [candidate]
                i = j

                sys.stdout.write("\rCandidates %6.2f%%" % ((100 * i) / float(limit)))
                sys.stdout.flush()

            if i > limit:
                break

        print("\rCandidates 100.00%")

    return inputs, references, candidates


def load_dev(limit=2900):
    print("Loading training data")
    return read(DEV_EN, DEV_DE, DEV_BEST, limit)


def load_test(limit):
    print("Loading testing data")
    return read(TEST_EN, TEST_DE, TEST_BEST, limit)


def parse_candidate(line):
    split_line = line.split(' ||| ')

    # Parse an id as an integer
    i = int(split_line[0])

    # Parse a candidate translation (with alignments) into a sentence.
    segments_and_alignments = map(lambda s: s.strip(),
                                  re.split(r'\|(\d+\-\d+)\|', split_line[1]))
    segments = segments_and_alignments[0::2]
    target = ' '.join(segments)

    return i, target, split_line
