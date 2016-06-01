# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import pro
from nltk.translate.bleu_score import corpus_bleu
from operator import itemgetter
import numpy as np
import sys


def best_reranking(inputs, candidates, classifier, normalizer, pca):
    """
    Returns a list of the best sentences according to the reranking
    Only the top sentence is returned
    """
    sentences = []

    for i, input in enumerate(inputs):

        results = []

        for j in range(0, len(candidates)):

            candidate = candidates[i][j]

            (_, target,candidate_features) = candidate
            (_, input_features) = input

            feature_vector = [np.concatenate((np.array(input_features), np.array(candidate_features)))]

            if normalizer is not None:
                feature_vector = normalizer.transform(feature_vector)

            if pca is not None:
                feature_vector = pca.transform(feature_vector)

            score = np.dot(classifier.coef_, np.transpose(feature_vector))[0][0]

            result = (score, target)
            results.append(result)

        (_, target) = sorted(results, key=itemgetter(0), reverse=True)[0]
        sentences.append(target)

    print("Reranking finished")

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


def print_evaluation(inputs, references, candidates, classifier, normalizer, pca):
    bleu_references = [[x] for x in references]
    bleu_hypotheses_baseline = best_baseline(inputs, candidates)

    baseline_blue = corpus_bleu(bleu_references, bleu_hypotheses_baseline)
    print("Baseline BLEU: %0.10f" % baseline_blue)

    bleu_hypotheses_reranking = best_reranking(inputs, candidates, classifier, normalizer, pca)

    if len(bleu_references) == len(bleu_hypotheses_reranking):
        reranking_blue = corpus_bleu(bleu_references, bleu_hypotheses_reranking)
        print("Reranking BLEU: %0.10f" % reranking_blue)

        blue_diff = reranking_blue - baseline_blue
        print("BLEU Diff: %0.10f" % blue_diff)
