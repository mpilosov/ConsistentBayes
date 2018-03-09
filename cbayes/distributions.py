## Copyright (C) 2018 Michael Pilosov

import numpy as np
import scipy.stats as sstats
from sklearn.neighbors.kde import KernelDensity

r"""
This module defines supported distributions and associated utility functions.
They are as follows:
    :class:`cbayes.distributions.gkde` (needs test)
    :class:`cbayes.distributions.parametric_dist` (needs development) 
    :method:`cbayes.distributions.supported_distributions` (tested)
    :method:`cbayes.distributions.assign_dist` (tested)
"""

def supported_distributions(distname=None):
    r"""
    TODO flesh out description.
    currently supports 'normal' and 'uniform'
    
    :param string distname: Name of a supported distributions. 
    If None, returns dictionary of accepted keys.
    
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
        'beta': sstats.beta, 
        'gamma': sstats.gamma
        }
    
    # The following overloads supported keys into our dictionary of distributions.
    if distname is not None: 
        if distname.lower() in ['gaussian', 'gauss', 'normal', 'norm', 'n']:
            distname = 'normal'
        elif distname.lower() in  ['uniform', 'uni', 'u']:
            distname = 'uniform'
        elif distname.lower() in ['chi2', 'c2', 'chisquared', 'chi_squared']:
            distname = 'chi2'
        elif distname.lower() in ['gamma', 'g', 'gam', 'gm']:
            distname = 'gamma'
        elif distname.lower() in ['beta', 'bt', 'b']:
            distname = 'beta'
        try:
            return D[distname]
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
        self.kde_object = sstats.gaussian_kde( data.transpose() )
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

class skde(object):
    r"""
    
    Custom wrapper around `sklearn.neighbors.kde.KernelDensity` to conform
    to our prefered syntax calling (following scipy conventions)

    """

    def __init__(self, data, mirror=False, **kwds):
        self.mirror = mirror
        if kwds is None:
            if self.mirror:
                self.kde_object = KernelDensity(kernel='gaussian').fit( np.vstack([-data, data]) )
            else:
                self.kde_object = KernelDensity(kernel='gaussian').fit(data)
        else:
            if self.mirror:
                self.kde_object = KernelDensity(**kwds).fit( np.vstack([-data, data]) )
            else:
                self.kde_object = KernelDensity(**kwds).fit(data)
        try:
            self.d = data.shape[1]
        except IndexError:
            self.d = 1
        self.n = data.shape[0]
        
    
    def rvs(self, size=1):
        r"""
        Generates random variables from a kde object. Wrapper function for 
        `sklearn.neighbors.kde.KernelDensity.sample`.
        
        :param int size: number of random samples to generate
        :param tuple size: number of samples is taken to be the first argument
        """
        if type(size) is tuple: 
            size=size[0]
                        
        if self.mirror: # have to generate twice as many samples
            num_samps = 0
            samps = []
            while num_samps < size:
                samp_proposal = self.kde_object.sample()
                if samp_proposal > 0:
                    samps.append(samp_proposal)
                    num_samps += 1
            samps = np.array(samps).reshape(size,self.d)
        else:
            samps = self.kde_object.sample(size)
            
        return samps
        #TODO write a test that makes sure this returns the correct shape
    
    def pdf(self, eval_points):
        r"""
        Generates random variables from a kde object. Wrapper function for 
        `sklearn.neighbors.kde.KernelDensity.score_samples`.
        
        :param eval_points: points on which to evaluate the density.
        :type eval_points: :class:`numpy.ndarray` of shape (num, dim)
        """
        
        #: TODO write a test that makes sure this returns the correct shape
        num_samples = eval_points.shape[0]
        if self.mirror:
            p = 2*np.exp( self.kde_object.score_samples( eval_points ) ).reshape(num_samples)
        else:
            try:
                p = np.exp( self.kde_object.score_samples( eval_points ) )
            except ValueError:
                p = np.exp( self.kde_object.score_samples( eval_points.reshape(-1,1) ) )
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
        
        
    def rvs(self, size=None):
        r"""
        TODO: Add this.
        """
        D = self.distributions
        if size is None:
            size = len(D.keys())
            
        if type(size) is tuple:
            assert(size[1] == len(D)) # make sure the dimensions are correc
            n = size[0]
        else:
            n = size

        for dist in D.keys():
            try:
                assert(D[dist] is not None)
            except AssertionError:
                raise(ValueError("""
                You are missing a distributionin key:%s, please use `self.setdist`"""%dist))
                      
        output = np.concatenate( [ D[dist].rvs(size=(n,1)) for dist in D.keys() ], axis=1)
        return output
    
    def pdf(self, eval_points):
        size = eval_points.shape
        D = self.distributions
        try:
            dim = size[1]
        except IndexError:
            dim = 1 
            if len(D) != dim:
                raise(IndexError("Could not infer dimensions. `eval_points` has the wrong shape."))
        n = size[0]
        eval_points = eval_points.reshape(n, dim)
        output = np.ones(n)
        for ind, dist in enumerate(D.keys()):
            try:
                assert(D[dist] is not None)
            except AssertionError:
                raise(ValueError("""
                You are missing a distributionin key:%s, please use `self.setdist`"""%dist))    
            output *= D[dist].pdf( eval_points[:,ind] )
        return output

    def evaluate(self, eval_points):
        return self.pdf(eval_points)

    def fit(self, dim):
        pass
    
    def mean(self):
        D = self.distributions
        
        for dist in D.keys():
            try:
                assert(D[dist] is not None)
            except AssertionError:
                raise(ValueError("""
                You are missing a distributionin key:%s, please use `self.setdist`"""%dist))
                      
        return [ D[dist].mean() for dist in D.keys() ]
    
    def std(self):
        D = self.distributions
        
        for dist in D.keys():
            try:
                assert(D[dist] is not None)
            except AssertionError:
                raise(ValueError("""
                You are missing a distributionin key:%s, please use `self.setdist`"""%dist))
                      
        return [ D[dist].std() for dist in D.keys() ]
        
        
    def assign_dist(self, dim, dist='normal', kwds=None):
        D = self.distributions
        if kwds is not None:
            D[str(dim)] = assign_dist(dist, **kwds)
        else:
            D[str(dim)] = assign_dist(dist)
        pass
    
    def set_dist(self, dim, dist='normal', kwds=None):
        D = self.distributions
        if kwds is not None:
            D[str(dim)] = assign_dist(dist, **kwds)
        else:
            D[str(dim)] = assign_dist(dist)
        pass

    def args(self): 
        r"""
        TODO: Add this.
        """
        pass
