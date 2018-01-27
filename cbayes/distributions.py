## Copyright (C) 2018 Michael Pilosov

from numpy import newaxis as np_newaxis
import scipy.stats as sstats
from scipy.stats import gaussian_kde


r"""
This module defines supported distributions and associated utility functions.
They are as follows:
    :class:`cbayes.distributions.gkde` (needs test)
    :class:`cbayes.distributions.parametric_dist` (needs development) 
    :method:`cbayes.distributions.supported_distributions` (tested)
    :method:`cbayes.distributions.assign_dist` (tested)
"""

def supported_distributions(d=None):
    r"""
    TODO flesh out description.
    currently supports 'normal' and 'uniform'
    
    rtype: :class:`scipy.stats._distn_infrastructure`
    :returns: scipy distribution object 
    
    rtype: :dict:
    :returns: dictionary with supported types.
    """
    # 
    # both take kwags `loc` and `scale` of type `numpy.ndarray` or `list`
    # method `sample_set.set_dist` just creates a handle for the chosen distribution. The longer of 
    # `loc` and `scale` is then inferred to be the dimension, which is written to sample_set.dim

    #: DICTIONARY OF SUPPORTED DISTRIBUTIONS:
    D = {
        'normal': sstats.norm, 
        'uniform': sstats.uniform,
        'chi2': sstats.chi2,
        }
    
    # The following overloads supported keys into our dictionary of distributions.
    if d is not None: 
        if d.lower() in ['gaussian', 'gauss', 'normal', 'norm', 'n']:
            d = 'normal'
        elif d.lower() in  ['uniform', 'uni', 'u']:
            d = 'uniform'
        elif d.lower() in ['chi2', 'c2', 'chisquared', 'chi_squared']:
            d = 'chi2'
        try:
            return D.get(d)
        except KeyError:
            print('Please specify a supported distribution. Type `?supported_distributions`')
    else: # if d is unspecified, simply return the dictionary.
        return D

def assign_dist(distribution, **kwds):
    r"""
    TODO clean up description of how this is overloaded.
    If a string is passed, it will be matched against the options for `supported_distributions`
    attach the scipy.stats._continuous_distns class to our sample set object
    
    rtype: :class:`scipy.stats._distn_infrastructure`
    :returns: scipy distribution object 
    """
    if type(distribution) is str:
        distribution = supported_distributions(distribution)
    if kwds is not None:
        return distribution(**kwds)
    else:
        return distribution

class gkde(object):
    r"""
    
    Custom wrapper around `scipy.stats.gaussian_kde` to conform
    to our prefered size indexing of (num, dim). 

    """

    def __init__(self, data):
        self.kde_object = gaussian_kde( data.transpose() )
        #: This is the primary difference
        self.d = self.kde_object.d
        self.n = self.kde_object.n

    def rvs(self, size=1):
        r"""
        Generates random variables from a kde object. Wrapper function for 
        `scipy.stats.gaussian_kde.resample`.
        
        :param int size: number of random samples to generate
        :param tuple size: number of samples is taken to be the first argument
        """
        if type(size) is tuple: 
            size=size[0]
        return self.kde_object.resample(size).transpose()
        #TODO write a test that makes sure this returns the correct shape
    
    def pdf(self, eval_points):
        r"""
        Generates random variables from a kde object. Wrapper function for 
        `scipy.stats.gaussian_kde.pdf`.
        
        :param eval_points: points on which to evaluate the density.
        :type eval_points: :class:`numpy.ndarray` of shape (num, dim)
        """
        
        #: TODO write a test that makes sure this returns the correct shape
        num_samples = eval_points.shape[0]
        p = self.kde_object.pdf( eval_points.transpose() ) 
        return p
    
class parametric_dist(object): 
    r"""
    
    TODO: add description. 
    TODO: add actual math. this is supposed to mimick scipy.stats, 
        except generalized to arbitrary mixtures, using familiar syntax 
        that hides the complexity.
        
    """
    def __init__(self, dim):
        self.dim = dim # this mimicks the scipy.stats.multivariate attribute
        self.distributions = {str(d): None for d in range(dim)}
        
    def rvs(self, size = None):
        r"""
        TODO: Add this.
        """
        if size is None: # if nothing specified, just generate one draw from the distribution of the RV
            size = (self.dim, 1)
        #TODO parse dict, hcat results.
        pass 

    def args(self): 
        r"""
        TODO: Add this.
        """
        pass
