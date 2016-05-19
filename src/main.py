#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement


import collections
import gzip
import nltk
import os
import pickle
import plf_parser
import spacy


DATA_DIR         = os.path.abspath(
                    os.path.join(os.path.dirname(__file__),'..','data'))
BASELINE_WEIGHTS = os.path.join(DATA_DIR,'baseline.weights')
DEV_BEST         = os.path.join(DATA_DIR,'nlp2-dev.1000best.gz')
DEV_BEST_PARTS   = [os.path.join(DATA_DIR,'nlp2-dev.1000best.gz'+part)
                    for part in ['.aa','.ab','.ac']]
DEV_DE           = os.path.join(DATA_DIR,'nlp2-dev.de.gz')
DEV_EN_PW        = os.path.join(DATA_DIR,'nlp2-dev.en.pw.plf-100.gz')
DEV_EN_S         = os.path.join(DATA_DIR,'nlp2-dev.en.s.gz')
TEST_BEST        = os.path.join(DATA_DIR,'nlp2-test.1000best.gz')
TEST_BEST_PARTS  = [os.path.join(DATA_DIR,'nlp2-test.1000best.gz'+part)
                    for part in ['.aa','.ab']]
TEST_DE          = os.path.join(DATA_DIR,'nlp2-test.de.gz')
TEST_EN_PW       = os.path.join(DATA_DIR,'nlp2-test.en.pw.plf-100.gz')
TEST_EN_S        = os.path.join(DATA_DIR,'nlp2-test.en.s.gz')


print("Loading SpaCy...")
en_nlp = spacy.load('en')
de_nlp = spacy.load('de')


en_s = u'Hello, world. Here are two sentences.'
de_s = u'Ich bin ein Berliner.'


doc = de_nlp(de_s)
for tok in doc:
    print(tok.vector)


def pos_feature(s,nlp):
    """
    Compute the POS feature vector given a sentence and an instance of spaCy.
    The POS feature vector is a vector which indicates, per POS-tag of the
    language, what ratio of the words in the sentence have this POS-tag.

    s  : input sentence
    nlp: instance of spaCy nlp
    """
    doc       = nlp(s,tag=True,parse=False,entity=False)
    pos_count = collections.Counter([tok.tag_ for tok in doc])
    return map(lambda tag: pos_count[tag] / len(doc), nlp.tagger.tag_names)


def BLEU(reference,candidate,n):
    """
    Compute the BLEU score for a given candidate sentence, with respect to a
    given reference sentence.

    reference: the reference translation
    candidate: the candidate translation
    n        : the size of the ngrams
    """
    return float(
        nltk.translate.bleu_score.modified_precision([reference],candidate,n=n))
