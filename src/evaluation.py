# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import pro
from nltk.translate.bleu_score import corpus_bleu


def best_reranking(inputs, candidates, classifier):
    """
    Returns a list of the best sentences according to the reranking
    Only the top sentence is returned
    """
    sentences = []

    for i, input in enumerate(inputs):

        def compare(candidate1, candidate2):
            if classifier.predict([pro.feature_vector(input, candidate1, candidate2)]) == [0]:
                return -1
            else:
                return 1

        (_, target, _) = sorted(candidates[i], cmp=compare)[0]
        sentences.append(target)

    return sentences


def best_baseline(inputs, candidates):
    """
    Returns a list of the best sentences according to the baseline, i.e. the phrase based model
    """
    sentences = []

    for i, input in enumerate(inputs):
        (_, target, _) = candidates[i][0]
        sentences.append(target)

    return sentences


def print_evaluation(inputs, references, candidates, classifier):
    bleu_references = [[x] for x in references]
    bleu_hypotheses_baseline = best_baseline(inputs, candidates)
    bleu_hypotheses_reranking = best_reranking(inputs, candidates, classifier)

    baseline_blue = corpus_bleu(bleu_references, bleu_hypotheses_baseline)
    print("Baseline BLEU: " + str(baseline_blue))
    reranking_blue = corpus_bleu(bleu_references, bleu_hypotheses_reranking)
    print("Reranking BLEU: " + str(reranking_blue))