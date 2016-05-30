# coding: utf-8

from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import pro
from nltk.translate.bleu_score import corpus_bleu
from operator import itemgetter
import sys


def best_reranking(inputs, candidates, references, classifier, normalizer, pca, limit=1000, improve_limit=10):
    """
    Returns a list of the best sentences according to the reranking
    Only the top sentence is returned
    """
    sentences = []
    classifications = 0

    if limit < improve_limit / 4:
        improve_limit = int(limit / 4)

    for i, input in enumerate(inputs):

        if i > 0:
            baseline_blue = corpus_bleu(references[:i], best_baseline(inputs[:i], candidates[:i]))
            reranking_blue = corpus_bleu(references[:i], sentences)
            blue_diff = reranking_blue - baseline_blue
            sys.stdout.write("\rReranking %6.2f%% classifications: %d current BLEU diff: %0.5f" %
                             ((100 * i) / float(len(inputs)), classifications, blue_diff))
            sys.stdout.flush()
        else:
            sys.stdout.write("\rReranking %6.2f%% classifications: %d" % ((100 * i) / float(len(inputs)), classifications))
            sys.stdout.flush()

        results, classifications_ = best_reranking_sentence(candidates[i], input, classifier, normalizer, pca, limit, improve_limit)
        classifications += classifications_

        (_, _, target) = results[0]
        sentences.append(target)

    print("\rReranking 100%% classifications: %d" % classifications)

    return sentences


def best_reranking_sentence(candidates, input, classifier, normalizer, pca, limit=1000, improve_limit=10):

    results = []
    classifications = 0

    for j in range(0, min(limit, len(candidates))):

        score = 0
        candidate1 = candidates[j]

        for k in range(0, min(limit, len(candidates))):
            if k != j:
                candidate2 = candidates[k]
                classifications += 1

                feature_vector = [pro.feature_vector(input, candidate1, candidate2)]

                if normalizer is not None:
                    feature_vector = normalizer.transform(feature_vector)

                if pca is not None:
                    feature_vector = pca.transform(feature_vector)

                if classifier.predict(feature_vector) != [-1]:
                    score += 1

        result = (score, j, candidate1[1])
        results.append(result)

    results = sorted(results, key=itemgetter(0), reverse=True)
    best_score = results[0][0]
    f_candidates = []

    for k, result in enumerate(results):
        (score, j, _) = result
        if k > improve_limit and score < best_score:
            break
        else:
            f_candidates.append(candidates[j])

    if f_candidates > 1 and improve_limit > 1:
        results, classifications_ = best_reranking_sentence(f_candidates, input, classifier, normalizer, pca, limit, max(1, int(improve_limit / 2)))
        classifications += classifications_

    return results, classifications


def best_baseline(inputs, candidates):
    """
    Returns a list of the best sentences according to the baseline, i.e. the phrase based model
    """
    sentences = []

    for i, input in enumerate(inputs):
        (_, target, _) = candidates[i][0]
        sentences.append(target)

    return sentences


def print_evaluation(inputs, references, candidates, classifier, normalizer, pca, limit=1000):
    bleu_references = [[x] for x in references]
    bleu_hypotheses_baseline = best_baseline(inputs, candidates)

    baseline_blue = corpus_bleu(bleu_references, bleu_hypotheses_baseline)
    print("Baseline BLEU: %0.10f" % baseline_blue)

    bleu_hypotheses_reranking = best_reranking(inputs, candidates, bleu_references, classifier, normalizer, pca, limit)
    reranking_blue = corpus_bleu(bleu_references, bleu_hypotheses_reranking)
    print("Reranking BLEU: %0.10f" % reranking_blue)

    blue_diff = reranking_blue - baseline_blue
    print("BLEU Diff: %0.10f" % blue_diff)
