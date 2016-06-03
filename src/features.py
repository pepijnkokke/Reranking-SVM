# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from nltk.util import ngrams
import itertools
import collections
import spacy
import re
import math


_en_nlp = 0
_de_nlp = 0


def input_features(line, params=(False, False, False, False, False)):
    """
    Parse a line from the inputs file into a input sentence and a feature vector
    """

    source = line.strip(' \t\n\r')
    decoded_source = source.decode('utf-8')

    feature_vector = []

    (include_pos, extended_pos, include_bigrams, include_embedding, _) = params

    if include_pos:
        pos_vector     = pos_feature(decoded_source, en_nlp(), simple_pos=not extended_pos)
        feature_vector = feature_vector + pos_vector
    if include_bigrams:
        bigrams_vector = pos_feature(decoded_source, en_nlp(), n=2)
        feature_vector = feature_vector + bigrams_vector
    if include_embedding:
        embedding      = en_nlp()(decoded_source).vector.tolist()
        feature_vector = feature_vector + embedding

    return source, feature_vector


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def candidate_features(line, params=(False, False, False, False, False)):
    """
    Parse a line from the candidate file into a target sentence and a feature vector
    """

    (i, target, feature_map, score) = candidate(line)
    decoded_target = target.decode('utf-8')

    features_from_map = sum(feature_map.values(), [])
    feature_vector = [score] + features_from_map

    (include_pos, extended_pos, include_bigrams, include_embedding, include_combinations) = params

    if include_combinations:
        per_tau = feature_map['PermutationExpectedKendallTau0']
        target_lm = feature_map['TargetLM'][0]
        source_lm = feature_map['SourceLM'][0]
        tm0 = feature_map['TranslationModel0']
        per_tau = [sigmoid(x) for x in per_tau]
        tm0 = reduce(lambda x, y: x * y, tm0)
        feature_vector = feature_vector + [target_lm ** 2, source_lm ** 2, tm0] + per_tau
    if include_pos:
        pos_vector = pos_feature(decoded_target, de_nlp(), simple_pos=not extended_pos)
        feature_vector = feature_vector + pos_vector
    if include_bigrams:
        bigrams_vector = pos_feature(decoded_target, de_nlp(), n=2)
        feature_vector = feature_vector + bigrams_vector
    if include_embedding:
        embedding = de_nlp()(decoded_target).vector.tolist()
        feature_vector = feature_vector + embedding

    return i, target, feature_vector


def candidate(candidate):
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

    i, target, split = candidate
    _, segments_and_alignments, feature_vector, score, alignments, source = split

    # source = source.strip(' \t\n\r')

    # Parse a candidate translation (with alignments) into a sentence.
    # segments_and_alignments = map(lambda s: s.strip(),
    #                               re.split(r'\|(\d+\-\d+)\|', segments_and_alignments))
    # segments = segments_and_alignments[0::2]
    # target = ' '.join(segments)

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


def en_nlp():
    global _en_nlp

    if _en_nlp == 0:
        _en_nlp = spacy.load('en')
        print('English language model loaded')

    return _en_nlp


def de_nlp():
    global _de_nlp

    if _de_nlp == 0:
        _de_nlp = spacy.load('de')
        print('German language model loaded')

    return _de_nlp


def pos_feature(s, nlp, n=1, simple_pos=True):
    """
    Compute the POS feature vector given a sentence and an instance of spaCy.
    The POS feature vector is a vector which indicates, per POS-tag of the
    language, what ratio of the words in the sentence have this POS-tag.

    s  : input sentence
    nlp: instance of spaCy nlp
    n  : the size of the n-grams over which the vector is built
    """
    doc = nlp(s, tag=True, parse=False, entity=False)

    # Compute the PoS-tags using spaCy.
    if simple_pos:
        pos_tags = [tok.pos_ for tok in doc]
        pos_sible = spacy.parts_of_speech.NAMES.values()
    else:
        pos_tags = [tok.tag_ for tok in doc]
        pos_sible = nlp.tagger.tag_names

    # Compute the n-grams of the PoS-tags.
    pos_tags = list(ngrams(pos_tags, n))
    pos_sible = itertools.combinations(pos_sible, n)

    pos_count = collections.Counter(pos_tags)

    def scalar(tag):
        return pos_count[tag]

    pos_count = map(lambda tag: scalar(tag), pos_sible)
    return pos_count
