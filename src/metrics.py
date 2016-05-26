# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from nltk.translate.bleu_score import modified_precision


def bleu(reference, candidate, n=4):
    """
    Compute the BLEU score for a given candidate sentence, with respect to a
    given reference sentence.

    reference: the reference translation
    candidate: the candidate translation
    """
    try:
        return float(modified_precision([reference], candidate, n=n))
    except ZeroDivisionError as error:
        return 0
