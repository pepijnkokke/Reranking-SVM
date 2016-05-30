# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import os
import re
import features
import msgpack
import sys

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

OUT_DIR = os.path.abspath(os.path.join('..', 'out'))

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def read(input_file, reference_file, candidates_file, limit):
    features.en_nlp()

    with open(input_file, 'r') as f:
        inputs = []
        for i in range(0, limit):
            if i % 100:
                sys.stdout.write("\rInputs %6.2f%%" % ((100 * i) / float(limit)))
                sys.stdout.flush()
            inputs.append(parse_input(f.readline()))
        print("\rInputs 100.00%")

    with open(reference_file, 'r') as f:
        references = [f.readline().strip(' \t\n\r') for i in range(0, limit)]

    features.de_nlp()

    with open(candidates_file, 'r') as f:
        candidates = []
        candidate_set = []
        i = 0

        while True:
            candidate = parse_candidate(f.readline())

            if candidate[0] == i:
                candidate_set.append(candidate)
            else:
                candidates.append(candidate_set)
                candidate_set = [candidate]
                i = candidate[0]

                sys.stdout.write("\rCandidates %6.2f%%" % ((100 * i) / float(limit)))
                sys.stdout.flush()

            if i > limit:
                break

        print("\rCandidates 100.00%")

    return inputs, references, candidates


def load_dev(limit=2900):
    dump = os.path.join(OUT_DIR, 'raw-dev-%d.out' % limit)
    if os.path.isfile(dump):
        with open(dump, 'r') as stream:
            print('Loading processed dev data')
            loaded = msgpack.unpack(stream, use_list=False)
    else:
        print('Reading and processing dev data')
        loaded = read(DEV_EN, DEV_DE, DEV_BEST, limit)
        with open(dump, 'w') as stream:
            print('Dumping processed dev data')
            msgpack.pack(loaded, stream)

    return loaded


def load_test(limit=2107):
    dump = os.path.join(OUT_DIR, 'raw-test-%d.out' % limit)
    if os.path.isfile(dump):
        with open(dump, 'r') as stream:
            print('Loading processed test data')
            loaded = msgpack.unpack(stream, use_list=False)
    else:
        print('Reading and processing test data')
        loaded = read(TEST_EN, TEST_DE, TEST_BEST, limit)
        with open(dump, 'w') as stream:
            print('Dumping processed test data')
            msgpack.pack(loaded, stream)

    return loaded


def parse_input(line):
    """
    Parse a line from the inputs file into a input sentence and a feature vector
    """

    source = line.strip(' \t\n\r')
    decoded_source = source.decode('utf-8')

    pos_features = features.pos_feature(decoded_source, features.en_nlp(), False)
    pos_bigram_features = features.pos_feature(decoded_source, features.en_nlp(), n=2)
    representation_feature = features.en_nlp()(decoded_source).vector.tolist()

    feature_vector = pos_features + pos_bigram_features + representation_feature

    return source, feature_vector


def parse_candidate(line):
    """
    Parse a line from the candidate file into a target sentence and a feature vector
    """

    (i, target, feature_map, score) = parse_candidate_line(line)
    decoded_target = target.decode('utf-8')

    features_from_map = sum(feature_map.values(), [])
    pos_features = features.pos_feature(decoded_target, features.de_nlp(), False)
    pos_bigram_features = features.pos_feature(decoded_target, features.de_nlp(), n=2)
    representation_feature = features.de_nlp()(decoded_target).vector.tolist()

    feature_vector = [score] + features_from_map + pos_features + pos_bigram_features + representation_feature

    return i, target, feature_vector


def parse_candidate_line(line):
    """
    Parse a candidate translation (a line from the 1000-best files) into
    a tuple containing (in order):

        i:              the 0-based sentence id           (int)
        source:         the source sentence               (str)
        target:         the translated sentence           (str)
        segments:       the segments and their alignments (list[(str,(int,int))])
        feature_vector: the feature vector                ({str: list[float]})
        score:          the score assigned by MOSES       (float)
        alignments:     the alignments                    ([(int,int)])

    Note: alignments in the "segments" field are pairs of states in the
    input lattice, whereas the alignments in the "alignments" field are
    pairs of a state in the input lattice together with the position of
    the output word.
    """
    i, segments_and_alignments, feature_vector, score, alignments, source = line.split(' ||| ')

    # source = source.strip(' \t\n\r')

    # Parse an id as an integer
    i = int(i)

    # Parse a candidate translation (with alignments) into a sentence.
    segments_and_alignments = map(lambda s: s.strip(),
                                  re.split(r'\|(\d+\-\d+)\|', segments_and_alignments))
    segments = segments_and_alignments[0::2]
    target = ' '.join(segments)

    # Parse a candidate translation (with alignments) into a list of segments.
    # segment_alignments = map(lambda s: tuple(map(int,s.strip().split('-'))),
    #                          segments_and_alignments[1::2])
    # segments = zip(segments,segment_alignments)

    # Parse a feature vector string into a dictionary.
    feature_vector = re.split(r'([A-Za-z]+0?)=', feature_vector)
    feature_names = feature_vector[1::2]
    feature_values = map(lambda s: map(float, s.strip().split()), feature_vector[2::2])
    feature_map = dict(zip(feature_names, feature_values))

    # Parse a score as a float.
    score = float(score)

    # Parse an alignment string into a list of tuples.
    # alignments = map(lambda s: tuple(map(int,s.split('-'))), alignments.strip().split(' '))

    return i, target, feature_map, score
