# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def bleu(reference, candidate):
    """
    Compute the BLEU score for a given candidate sentence, with respect to a
    given reference sentence.

    reference: the reference translation
    candidate: the candidate translation
    """
    chen_cherry = SmoothingFunction()
    try:
        return sentence_bleu([reference], candidate, smoothing_function=chen_cherry.method7)
    except ZeroDivisionError as error:
        return 0
    except AttributeError as error:
        return 0
