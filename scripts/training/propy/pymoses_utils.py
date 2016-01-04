#!/usr/bin/env python3 -*- coding: utf-8 -*-
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

import re, io
from collections import defaultdict, namedtuple

import numpy as np

############################################################################
# Moses related manipulations 
############################################################################

moses_param_pattern = re.compile(r'''([^\s=]+)=\s*((?:[^\s=]+(?:\s|$))*)''')

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

def update_paramters(parameters, new_unrolled_params):
    """
    >>> new_weights = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
    >>> update_paramters(default_moses_params, new_weights)
    {'Distortion0': [0.01], 'WordPenalty0': [0.14], 'TranslationModel0': [0.1, 0.11, 0.12, 0.13], 'LM0': [0.02], 'PhrasePenalty0': [0.09], 'LexicalReordering0': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]}
    """
    i = 0
    for p in sorted(parameters):
        if p == 'UnknownWordPenalty0': # No change to unknown words parameter.
            continue
        for j, _ in enumerate(parameters[p]):
            parameters[p][j] = new_unrolled_params[i]
            i+=1
    return parameters

def params_to_string(parameters):
    """
    >>> new_weights = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
    >>> params_to_string(update_paramters(default_moses_params, new_weights))
    Distortion0=0.01
    LM0=0.02
    LexicalReordering0=0.03 0.04 0.05 0.06 0.07 0.08
    PhrasePenalty0=0.09
    TranslationModel0=0.1 0.11 0.12 0.13
    WordPenalty0=0.14
    """
    pstr = []
    for p in sorted(parameters):
        pstr.append(p + '=' + ' '.join(map(str,parameters[p])))
    return "\n".join(pstr)

def overwrite_mosesini_weights(old_mosesini_file, new_moses_ini_file,
                               new_weights):
    old_weights = ''
    with open(old_mosesini_file, 'r') as fin, open(new_moses_ini_file, 'w') as fout:
        for line in fin:
            fout.write(line)
            if line.startswith('[weight]'):
                break
        for line in fin:
            old_weights+=line.strip() + ' '
        old_weights = parse_parameters(old_weights,to_unroll=False)
        param_str = params_to_string(update_paramters(old_weights, new_weights))
        fout.write(param_str)

############################################################################
# Moses I/O reading.
############################################################################        
        
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

def read_plaintext(infile): 
    with io.open(infile, 'r', encoding='utf8') as fin:
        for line in fin:
            yield line.strip()

############################################################################
# Metrics I/O reading.
############################################################################        

