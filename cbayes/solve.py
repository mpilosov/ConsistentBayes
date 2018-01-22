#!/home/mpilosov/anaconda3/envs/py3/bin/python
## Copyright (C) 2018 Michael Pilosov

r"""
This module contains the methods for solving the stochastic inverse problem:
    :method:`cbayes.solve.compute_ratio`
    :method:`bet.sample.discretization`
    :class:`bet.sample.length_not_matching`
    :class:`bet.sample.dim_not_matching`
"""

import cbayes.sample as samp

#: TODO add logging/warnings, save options, load options.
# import os, logging 

def compute_ratio(problem_set):
    r"""
    TODO: Add description
    :param sample_set: 
    :type sample_set: :class:`~/cbayes.sample.sample_set`
    
    :rtype: :class:`numpy.ndarray` of shape(num,)
    :returns: ratio of observed to pushforward density evaluations
    """
    data = problem_set.output.samples
    ratio = problem_set.observed_dist.pdf(data) / problem_set.pushforward_dist.pdf(data)
    ratio = ratio.ravel()
    problem_set.ratio = ratio
    return ratio

def perform_accept_reject(problem_set, seed=None):
    r"""
    Perform a standard accept/reject procedure by comparing 
    normalized density values to draws from Uniform[0,1]
    
    :param sample_set: 
    :type sample_set: :class:`~/cbayes.sample.sample_set`
    
    :param int seed: Your seed for the accept/reject.
    
    It is encouraged that you run this multiple times when num_samples is small
    Then, average the results to get an average acceptance rate.
    """
    M = np.max(problem_set.ratio)
    eta_r = problem_set.ratio/M
    if seed is None:
        np.random.seed(problem_set.seed)
    else:
        np.random.seed(seed)
    problem_set.accept_inds = [i for i in range(problem_set.input.num_samples) 
                if eta_r[i] > np.random.rand() ] 
    pass

#: TODO ADD A LOT MORE METHODS. Weighted KDE, surrogate post, MCMC, etc.
