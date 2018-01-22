## Copyright (C) 2018 Michael Pilosov

# Michael Pilosov 01/21/2018


r"""
This module contains the methods for solving the stochastic inverse problem:
    :method:`cbayes.solve.perform_accept_reject`

"""

# import cbayes.sample as sample
import numpy as np
from nose import with_setup # optional

#: TODO add logging/warnings, save options, load options.
# import os, logging 

def perform_accept_reject(samples, ratios, seed=0):
    r"""
    TODO: CHECK SIZES!!! samples and ratios should match up.
    Perform a standard accept/reject procedure by comparing 
    normalized density values to draws from Uniform[0,1]
    
    :param samples: Your samples.
    :type samples: :class:`~/cbayes.sample.sample_set` of shape (num, dim)
    
    :param ratios:
    :type ratios: :class:`numpy.ndarray` of shape (num,)
    :param int seed: Your seed for the accept/reject.
    
    It is encouraged that you run this multiple times when num_samples is small
    Then, average the results to get an average acceptance rate.
    """
    num_samples = len(ratios)
    #: check dimensions here.
    
    #: normalize the ratios to [0, 1]
    M = np.max(ratios)
    eta_r = ratios/M
    
    np.random.seed(seed)
    accept_inds = [i for i in range(num_samples) 
                if eta_r[i] > np.random.rand() ] 
    return accept_inds

#: TODO ADD A LOT MORE METHODS. Weighted KDE, surrogate post, MCMC, etc.

def problem(problem_set, method='AR', seed=0):
    r"""
    This solves the inverse problem. It's a wrapper for other functions.
    
    :param problem_set: Your problem_set.
    :type problem_set: :class:`~/cbayes.sample.problem_set` 

    :param str method: One of the supported methods ('AR' for accept/reject)
    """
    samples = problem_set.input.samples 
    if problem_set.ratio is None:
        assert ValueError("ratios not set")
    ratios = problem_set.ratio
    
    if method == 'AR':
        accept_inds = perform_accept_reject(samples, ratios, seed)
    else:
        raise TypeError("method given not supported. Please see documentation.")
    
    problem_set.accept_inds = accept_inds
    pass
