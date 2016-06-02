# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

from time import time

from nltk.translate.bleu_score import corpus_bleu
from operator import itemgetter
import numpy as np
import sys
import features


def best_reranking(inputs, candidates, classifier, normalizer, pca, params):
    """
    Returns a list of the best sentences according to the reranking
    Only the top sentence is returned
    """
    t0 = time()

    sentences = []
    weights = np.array(classifier.coef_)

    for i, input in enumerate(inputs):

        sys.stdout.write("\rReranking %6.2f%%" % ((100 * i) / float(len(inputs))))
        sys.stdout.flush()

        results = []

        (_, input_features) = features.input_features(input, params)
        input_features = np.array(input_features)

        for candidate in candidates[i]:

            (_, target, candidate_features) = features.candidate_features(candidate, params)
            candidate_features = np.array(candidate_features)

            feature_vector = [np.concatenate((input_features, candidate_features))]

            if normalizer is not None:
                feature_vector = normalizer.transform(feature_vector)

            if pca is not None:
                feature_vector = pca.transform(feature_vector)

            score = np.dot(weights, np.transpose(feature_vector))[0][0]

            result = (score, target)
            results.append(result)

        (_, target) = sorted(results, key=itemgetter(0), reverse=True)[0]
        sentences.append(target)

    print("\rReranking 100.00%")

    reranking_time = (time() - t0)
    print("done in %0.3fs" % reranking_time)

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


def evaluation(data, classifier, normalizer, pca, params):

    (inputs, references, candidates) = data

    bleu_references = [[x] for x in references]
    bleu_hypotheses_baseline = best_baseline(inputs, candidates)

    baseline_blue = corpus_bleu(bleu_references, bleu_hypotheses_baseline)
    print("Baseline BLEU: %0.10f" % baseline_blue)

    bleu_hypotheses_reranking = best_reranking(inputs, candidates, classifier, normalizer, pca, params)

    reranking_blue = corpus_bleu(bleu_references, bleu_hypotheses_reranking)
    print("Reranking BLEU: %0.10f" % reranking_blue)

    blue_diff = reranking_blue - baseline_blue
    print("BLEU Diff: %0.10f" % blue_diff)

    return baseline_blue, reranking_blue, blue_diff
