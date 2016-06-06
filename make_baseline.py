#!/usr/bin/env python

import re

with open('data/nlp2-test.1000best', 'r') as f:

    k_     = None
    score_ = None

    for ln in f:
        k, sentence, _, score, _, _ = ln.split(' ||| ')
        k        = int(k)
        sentence = ''.join(re.split(r'\|(\d+\-\d+)\|', sentence)[0::2])
        score    = float(score)

        if k == k_:
            if not score_ is None:
                assert score <= score_
        else:
            print sentence

        score_ = score
        k_     = k
