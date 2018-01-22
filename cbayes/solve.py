#!/home/mpilosov/anaconda3/envs/py3/bin/python
## Copyright (C) 2018 Michael Pilosov

"""
This module contains the methods for solving the stochastic inverse problem:
    :method:`cbayes.solve.compute_ratio`
    :method:`bet.sample.discretization`
    :class:`bet.sample.length_not_matching`
    :class:`bet.sample.dim_not_matching`
"""
import numpy as np
import scipy.stats as sstats
from scipy.stats import gaussian_kde
#: TODO add logging/warnings, save options, load options.
# import os, logging 


def compute_ratio(self):
    data = self.output.samples
    self.ratio = self.observed_dist.pdf(data) / self.pushforward_dist.pdf(data)
    self.ratio = self.ratio.ravel()
   

def perform_accept_reject(self, seed=None):
    """
    Perform a standard accept/reject procedure by comparing 
    normalized density values to draws from Uniform[0,1]
    
    :param int seed: Your seed for the accept/reject.
    
    It is encouraged that you run this multiple times when num_samples is small
    Then, average the results to get an average acceptance rate.
    """
    M = np.max(self.ratio)
    eta_r = self.ratio/M
    if seed is None:
        np.random.seed(self.seed)
    else:
        np.random.seed(seed)
    self.accept_inds = [i for i in range(self.input.num_samples) if eta_r[i] > np.random.rand() ] 

#: TODO ADD A LOT MORE METHODS. Weighted KDE, surrogate post, MCMC, etc.


