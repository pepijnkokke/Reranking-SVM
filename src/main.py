#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import gzip
import os
import pickle
import plf_parser


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
