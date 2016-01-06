#!/usr/bin/env python3 -*- coding: utf-8 -*-
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

"""PRO Tuning in Python.

This is a python implementation of PRO tuning using scikit-learn classifier(s).
This implementation is largely based on Chahuneau et al. (2012) pycdec parameter 
estimation code (from http://victor.chahuneau.fr/pub/pycdec/)

Usage:
  pro.py --config FILE --devsrc FILE --devtrg FILE [--output FILE]
  pro.py -f FILE -s FILE -t FILE -o FILE --nbest=1000 --jobs=1
  pro.py -f FILE -s FILE -t FILE -o FILE 
  
Options:
  -h --help       Show this screen.
  -f --config     Moses configuration file, i.e. moses.ini (Complusory)
  -s --devsrc     Plaintext file from the source language (Complusory)
  -t --devtrg     Plaintext file from the target language (Complusory)
  -o --output     Output file to save the tuned parameters
  --nbest         Size of the n-best list use to tune the system [default: 1000]
  --jobs          No. of threads to run run Moses in parallel [default" 1]
"""

from __future__ import print_function

import io, re, heapq, sys, os
from collections import defaultdict, namedtuple
import random

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from bleu import corpus_bleu, sentence_bleu_nbest, bleu_from_file
from moses import *

random.seed(0)

def random_integer(max_num):
    return random.randint(0, max_num -1)


def get_pairs(nbest_hypotheses, scores, n_samples=10000, n_pairs=10000, 
              score_threshold=0.05):
    num_hyp = len(nbest_hypotheses)
    # Use the uniform distribution to sample n random pairs
    # from the set of candidate translations
    pairs = []
    for _ in range(n_samples):
        rand1, rand2 = random_integer(num_hyp), random_integer(num_hyp)
        ci, cj = nbest_hypotheses[rand1], nbest_hypotheses[rand2] 
        if abs(scores[rand1] - scores[rand2]) < score_threshold: 
            continue
        # Find the differences in parameters/weights for this pair of sentence.
        x = ci.params - cj.params
        # Find the differences in score.
        y = scores[rand1] - scores[rand2]
        pairs.append((x,y))
    # From the potential pairs kept in the previous step,
    # keep the s pairs that have the highest score
    for x, y in heapq.nlargest(n_pairs, pairs, key=lambda xy: abs(xy[1])):
        # For each pair kept, make two data points
        yield x, y # In the positive space
        yield -1 * x, -1 * y # In the negative space


def pro_one_cycle(references, nbestlist, metric=sentence_bleu_nbest,
                  n_samples=5000, n_pairs=1000, regressor=LinearRegression, 
                  smoothing=1, epsilon=0.1, alpha=5, k=5, 
                  weights=(0.25, 0.25, 0.25, 0.25)):
    # The DictVectorizer converts dictionaries into sparse vectors
    vectorizer = DictVectorizer()
    # Find out the no. of parameters.
    num_params = len(nbestlist[0][0].params)
    # Collect training pairs
    X, Y = [], []
    for i, reference in enumerate(references):
        nbest_hypotheses = nbestlist[i]
        print('Reference sentence {}...'.format(i), file=sys.stderr) 
        scores = list(metric(reference, nbest_hypotheses, smoothing=smoothing))
        #print (nbest_hypotheses[0])
        #print (scores)
        for x, y in get_pairs(nbest_hypotheses, scores, n_samples, n_pairs):
            # Converts weights into a dictionary where 
            # key = weights ID ; value = difference in parameter weight.
            x = dict(zip(range(num_params), x))
            X.append(x)
            Y.append(y)

    # Train a linear regression model
    model = regressor()
    X = vectorizer.fit_transform(X)
    model.fit(X, Y)
    # Return the weights with the learned model
    return model.coef_

def pro_tuning(source_file, reference_file, config_file, moses_dir, 
               working_dir=os.getcwd(), n_iterations=100,
               max_char_span=1000, nbest_size=1000, threads=4,
               metric=sentence_bleu_nbest, 
               n_samples=5000, n_pairs=1000, regressor=LinearRegression, 
               smoothing=1, epsilon=0.1, alpha=5, k=5, 
               weights=(0.25, 0.25, 0.25, 0.25)):
    # Read the reference into memory
    # this will be the only constant in the whole process.
    references = list(read_plaintext(reference_file))
   
    # Setup the Moses wrapper.
    momo = MosesDecoder(moses_dir)
    # *new_config_file* stores the path to the latest config file after each
    # tuning iteration.
    new_config_file = config_file
    # Keeps track of best bleu score.
    bleu_scores = {-1: 0.0}
    
    # Star the tunig iteration.
    for num_iter in range(n_iterations):
        output_file = "tmpnoutput.run{}".format(num_iter)
        log_file = "tmpnlog.run{}".format(num_iter)
        # Create logfile name.
        # Decode the source_file and produce the output_file and nbestlist.
        output_file, nbest_file = momo.hiero_decode(source_file, output_file, 
                                    new_config_file, log_file, working_dir, 
                                    max_char_span, threads, nbest_size, 
                                    run_num=num_iter)
        """
        output_file, nbest_file = momo.phrase_decode(source_file, output_file, 
                                    new_config_file, log_file, working_dir,
                                    threads, nbest_size, run_num=num_iter)
        """
        # Reads nbestlist file output into a pythonic object.
        nbest_list = read_nbestlist(nbest_file)
        new_weights = pro_one_cycle(references, nbest_list, metric,
                                    n_samples, n_pairs, regressor, 
                                    smoothing, epsilon, alpha, k, weights)
        # Produce the ouput.
        new_config_file = "{}/moses-run{}.ini".format(working_dir, num_iter) 
        overwrite_mosesini_weights(config_file, new_config_file, new_weights)
        # Very very lazy way to get the BLEU scores of the new output.
        bleu_score_now = bleu_from_file(reference_file, output_file)
        # Tracks the BLEU score. 
        bleu_scores[num_iter] = bleu_score_now
        
        print('Run {}, BLEU: {} -> {}'.format(num_iter, bleu_scores[num_iter-1], 
                                              bleu_score_now), file=sys.stderr, end="")
        
        
nltk_translate = '/home/alvas/git/nltk/nltk/translate/'
mosesini_file = nltk_translate + 'mertfiles/moses.ini'
source_file = deven = nltk_translate + 'mertfiles/dev.en'
reference_file = devru = nltk_translate + 'mertfiles/dev.ru'
nbestlist_file = nbestlist_ru = nltk_translate + 'mertfiles/dev.100best.ru' 
moses_dir = '/home/alvas/mosesdecoder/'
sources = list(read_plaintext(source_file))
references = list(read_plaintext(reference_file))
nbestlist = read_nbestlist(nbestlist_file)


#new_weights = pro_one_cycle(references, nbestlist)
#print (new_weights)
#print (update_paramters(default_moses_params, new_weights))

moses_dir = '/home/alvas/mosesdecoder/'
source_file = devcs = '/home/alvas/cs2en_model/tuning-task-dev-v2/newstest2014-csen-ref.cs.toklc'
reference_file = deven = '/home/alvas/cs2en_model/tuning-task-dev-v2/newstest2014-csen-ref.en.toklc'
mosesini_file = '/home/alvas/cs2en_model/moses.ini'


pro_tuning(source_file, reference_file, mosesini_file, moses_dir, 
           working_dir='/home/alvas/cs2en_model/pro-testrun/',
           n_iterations=10, threads=1, nbest_size=10)


"""
[ 0.00095703  0.0011907   0.00157549 -0.00040763  0.01321348 -0.00279545
  0.00144725  0.00658409 -0.00168052  0.00530775 -0.00056132 -0.00145704
 -0.01289229]

"""