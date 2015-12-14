#!/usr/bin/env python3 -*- coding: utf-8 -*-
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

"""
This is a python implementation of PRO tuning using scikit-learn classifier(s).
"""

import io, re, heapq
from collections import defaultdict, namedtuple
from random import randint

import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from nltk.translate import bleu

moses_param_pattern = re.compile(r'''([^\s=]+)=\s*((?:[^\s=]+(?:\s|$))*)''')

def read_params_from_moses_ini(mosesinifile):
    parameters_string = ""
    for line in reversed(open(mosesinifile, 'r').readlines()):
        if line.startswith('[weight]'):
            return parse_parameters(parameters_string.strip())
        else:
            parameters_string+=line.strip() + ' ' 

def parse_parameters(parameters_string, to_unroll=True):
    """
    Convert a parameter string from the nbest list into a flatten list of floats.
    
    >>> param_str = "LexicalReordering0= -5.28136 0 0 -6.95799 0 0 Distortion0= 0 LM0= -172.048 WordPenalty0= -27 PhrasePenalty0= 26 TranslationModel0= -39.8547 -26.9387 -26.669 -20.8583"
    >>> parse_parameters(param_str)
    [0.0, -172.048, -5.28136, 0.0, 0.0, -6.95799, 0.0, 0.0, 26.0, -39.8547, -26.9387, -26.669, -20.8583, -27.0]
    """
    params = dict((k, list(map(float, v.split())))
                   for k, v in moses_param_pattern.findall(parameters_string))
    if to_unroll:
        return unroll_parameters(params)
    return params

def unroll_parameters(params):
    """ 
    Convert dict of list parameters into a single vector 
    
    >>> unroll_parameters(default_moses_params)
    array([0.3, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, -1.0])
    """
    param_vec = []
    for p in sorted(params):
        param_vec+=params[p]
    return np.array(param_vec)


def read_plaintext(infile): 
    with io.open(infile, 'r', encoding='utf8') as fin:
        for line in fin:
            yield line.strip()


# A `Hypothesis` is a light weight object that holds 
# (i) translation string and (ii) the unrolled parameter list
Hypothesis = namedtuple('Hypothesis', 'translation, params')

def read_nbestlist(nbestlist_file):
    nbestlist = defaultdict(list)
    for line in read_plaintext(nbestlist_file):
        sentid, translation, params, score = line.split(' ||| ')
        hyp = Hypothesis(translation, parse_parameters(params))
        nbestlist[int(sentid)].append(hyp)
    return nbestlist


def random_integer(max_num):
    return randint(0, max_num -1)


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

def nltk_bleu_scores(reference, nbest_hypotheses):
    return [bleu([reference], hyp.translation, weights=[0.25]*4) for 
            hyp in nbest_hypotheses]


def pro_one_cycle(references, nbestlist, metric=nltk_bleu_scores, 
               n_samples=5000, n_pairs=1000, regressor=LinearRegression):
    # The DictVectorizer converts dictionaries into sparse vectors
    vectorizer = DictVectorizer()
    num_params = len(nbestlist[0][0].params)
    # Collect training pairs
    X, Y = [], []
    for i, reference in enumerate(references):
        nbest_hypotheses = nbestlist[i] 
        scores = metric(reference, nbest_hypotheses)
        #print (nbest_hypotheses[0])
        #print (scores)
        for x, y in get_pairs(nbest_hypotheses, scores, n_samples, n_pairs):
            # Converts weights into a dictionary where 
            # key = weights ID ; value = difference in paramter weight.
            x = dict(zip(range(num_params), x))
            X.append(x)
            Y.append(y)

    # Train a linear regression model
    model = regressor()
    X = vectorizer.fit_transform(X)
    model.fit(X, Y)
    # Return the weights with the learned model
    return model.coef_
            
def pro_tuning(source_file, reference_file, moses_ini, n, moses_bin,
               metric=nltk_bleu_scores, n_iterations,
               n_samples=5000, n_pairs=1000, regressor=LinearRegression):
    
    moses_cmd = "{} -f {} -n-best-list tmpnbest {} < {}"
    references = list(read_plaintext(reference_file))
    
    for num_iter in range(n_iterations):
        # Generate the nbest-list file given the moses.ini 
        cmd = moses_cmd.format(moses_bin, moses_ini, n, source_file)
        os.system(cmd)
        # Read the nbest-list and references into python object.
        nbestlist = read_nbestlist('tmpnbest')
        # Retrieve the new set of parameters given the nbestlist.
        new_params = pro_one_cycle(references, nbestlist, metric, 
                                   n_samples, n_pairs, regressor)
        # Create a new moses.ini with new parameters.
        pass # Too tired to code now at 3.30am... BRB!
        
    
    



nltk_translate = '/home/alvas/git/nltk/nltk/translate/'
mosesinifile = nltk_translate + 'mertfiles/moses.ini'
source_file = deven = nltk_translate + 'mertfiles/dev.en'
reference_file = devru = nltk_translate + 'mertfiles/dev.ru'
nbestlist_file = nbestlist_ru = nltk_translate + 'mertfiles/dev.100best.ru' 

sources = list(read_plaintext(source_file))
references = list(read_plaintext(reference_file))
nbestlist = read_nbestlist(nbestlist_file)

ref_len = len(references)

pro_one_cycle(references, nbestlist)