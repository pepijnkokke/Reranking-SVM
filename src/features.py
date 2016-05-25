# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from nltk.util import ngrams
import itertools
import collections
import spacy

_en_nlp = 0
_de_nlp = 0


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
    pos_count = map(lambda tag: pos_count[tag] / len(pos_tags), pos_sible)
    return pos_count
