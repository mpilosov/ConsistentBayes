## Copyright (C) 2018 Michael Pilosov

# Michael Pilosov 01/21/2018


r"""
This module defines the data structure classes for ConsistentBayes. They are: 
    :class:`cbayes.sample.sample_set`
    :class:`cbayes.sample.problem_set`
"""

import numpy as np
import logging
import cbayes.distributions as distributions
# import cbayes.solve

#: this is for saving/loading. 
#: TODO add glob.glob() methods with os.path.dirname/pathname
#: import glob 
# import warnings (# what does warnings do that logging cannot?)

def map_samples_and_create_problem(input_sample_set, QoI_fun):
    r"""
    TODO: full description, check type conformity
    
    :param input_sample_set:
    :type input_sample_set: :class:`~cbayes.sample_set` input samples
    """
    # pass a QoI_fun, grab the input samples and map them to the data space.
    input_samples = input_sample_set.samples
    if input_samples is None:
        raise(AttributeError(" `input_sample_set.samples` cannot be None."))
    output_samples = QoI_fun(input_samples) # make sure your QoI_fun conforms to size (num_samples, dim)
    if len(output_samples.shape) == 1: # enfore shape parameters by force.
        output_samples = output_samples[:, np.newaxis]
    if output_samples.shape[0] != input_samples.shape[0]: # attempt to handle 1xn arrays as nx1: 
        logging.warn('Your model returns the wrong shape. Attempting transpose...')
        output_samples = output_samples.transpose()
    if output_samples.shape[0] != input_samples.shape[0]:
        raise(AssertionError(" Model provided does not conform to shape requirements. Please return (n,d) `numpy.ndarray`."))
    output_sample_set = sample_set(size=output_samples.shape)
    output_sample_set.samples = output_samples
    pset = problem_set(input_sample_set, output_sample_set)
    pset.model = QoI_fun
    return pset

class sample_set(object):
    def __init__(self, size=(None, None), seed=121):
        r"""

        Initialization
        
        :param size: Dimension of the space in which these samples reside.
        :type size: :class:`numpy.ndarray` of sample values of shape (num, dim)
        
        :param int seed: random number generator seed
        """
        # tuple `size` should be of format (num_samples, dim). 
        # Will write these attributes to class `sample_set`
        # If `size` is given as an integer, it is inferred to be dimension.
        if type(size) is tuple:
            self.num_samples = size[0]
            if len(size) == 1:
                self.dim = 1
            else:
                self.dim = size[1] # dimension
        elif type(size) is int:
            self.dim = size
            self.num_samples = None # used as a default. 
            # will infer/set `num_samples` from call to `generate_samples`
        else:
            logging.warning(" Please specify a valid size parameter. Defaulting to None.")
            self.dim = None
            self.num_samples = None
        #: dist TODO description
        if self.dim is not None:
            self.dist = distributions.parametric_dist(self.dim)
        else:
            self.dist = None
        #: :class:`numpy.ndarray` of samples of shape (num, dim)
        self.samples = None # this holds the actual samples we generate.
        #: :param int seed: random number generator seed
        self.seed = seed 
        
        #self.bounds = None # bounds on the space
        #self.weights = None # weights for weighted KDE. 
        # If samples taken from dist, should be set to 1/N. #TODO default this

  
    def set_dim(self, dimension=None):
        r"""
        TODO: Add this.
        """
        if self.dim is None:
            if dimension is not None:
                self.dim = int(abs(dimension))
            else:
                self.dim = 1 # default option is self.dim not yet set.
        elif self.dim is not None:
            if dimension is not None:
                self.dim = int(abs(dimension))
            # otherwise, if nothing specified, leave it alone.
        else:
            assert TypeError("Please specify an integer-valued `dimension` greater than zero.")
        self.dist = distributions.parametric_dist(self.dim)
        pass

    def set_num_samples(self, num_samples=None):
        r"""
        TODO: Add this.
        """
        if self.num_samples is None:
            if num_samples is not None:
                self.num_samples = int(abs(num_samples))
            else:
                self.num_samples = 1000 # default option is self.num_samples not yet set.
        elif self.num_samples is not None:
            if num_samples is not None:
                self.num_samples = int(abs(num_samples))
        else:
            assert TypeError("Please specify an integer-valued `num_samples` greater than zero.")
        pass
        
    def set_dist(self, distribution='uniform', kwds=None, dim=None):
        r"""
        TODO: Add this.
        """
        if (kwds is not None) and (dim is not None):
            self.dist.set_dist(dim, distribution, kwds)
        elif (kwds is None) and (dim is not None):
            self.dist.set_dist(dim, distribution)

        # Here we override the default errors printed by scipy.stats with our own.
        elif (kwds is None) and (distributions.supported_distributions(distribution) is 'chi2' ):
            raise(AttributeError("If you are using a chi2 distribution, please pass `df` as a key-value pair in a dictionary. Ex: {'df': 20}."))
        elif (kwds is None) and (distributions.supported_distributions(distribution) is 'beta'):
            raise(AttributeError("If you are using a Beta dist, please pass `a` and `b` as key-value pairs in a dictionary. Ex: {'a': 1, 'b': 1}"))
        # the following allows for manual overrides not using the parametric object.
        # e.g. kwds = {'loc': [1,2,3]}
        elif dim is None:
            logging.warn("INPUT: No dimension specified. You will be using `scipy.stats` for your distributions instead of the parametric object. Be warned that functions like `.pdf` may not work as expected.")
            if kwds is not None:
                self.dist = distributions.assign_dist(distribution, **kwds)
            else: 
                self.dist = distributions.assign_dist(distribution)
        pass
 
    def setup(self):
        r"""
        TODO: Add this.
        """
        # dummy function that runs the defaults to set up an unbounded 1D problem with gaussian prior.
        self.set_dim()
        self.set_num_samples()
        self.set_dist() 
        pass

    def generate_samples(self, num_samples=None, seed=None, verbose=True):
        r"""
        TODO: Add this.
        """
        #TODO check if dimensions specified, if not, prompt user.
        # Since we want this function to work by default, we temporarily set a default. TODO remove this behavior.
        if self.dim is None:
            if verbose: 
                print('Dimension unspecified. Assuming 1D')
            self.dim = 1
        if num_samples is not None:
            if verbose:
                logging.warning("Number of samples declared, written to `sample_set.num_samples`.")
            self.num_samples = num_samples
        else:
            if self.num_samples is None:
                if verbose:
                    logging.warning("Number of samples undeclared, choosing 1000 by default.")
                self.num_samples = 1000
        if seed is None:
            np.random.seed(self.seed) 
        else:
            np.random.seed(seed)
            self.seed = seed # store the last used random seed.
        self.samples = self.dist.rvs(size=(self.num_samples, self.dim))
        return self.samples

class problem_set(object):
    r"""
    TODO: Add this.
    """
    def __init__(self, input_set=None, output_set=None, seed=None):
        self.input = input_set
        self.output = output_set
        self.model = None
        self.prior_dist = self.input.dist
        self.pushforward_dist = self.output.dist # kde object. should have rvs functionality. TODO: double check sizing with test.
        self.posterior_dist = None # this will be the dictionary object which we can use with .rvs(num_samples)
        
        if self.output.dim is not None:
            self.observed_dist = distributions.parametric_dist(self.output.dim)
        else:
            self.observed_dist = None

        self.accept_inds = None # indices into input_sample_set object associated with accepted samples from accept/reject
        self.ratio = None # the ratio is the posterior density evaluated on the `input_set.samples`
        self.pf_pr_eval = None
        if seed is None:
            self.seed = 12112
        else:
            self.seed = seed


    def get_problem(self):
        r"""
        TODO: Add this.
        """
        if type(self.input.samples) is __main__.sample_set:
            print('Your input space is %d-dimensional'%(self.input.dim))
            print('\t and is (%d, %d)'%(self.input.samples.shape))
       
            if type(self.output.samples) is __main__.sample_set:
                #TODO overload just a set of evaluated samples as ndarray, determine attributes and write to new output_samples
                print('Your output space is %d-dimensional'%(self.output.dim))
                print('\t and is (%d, %d)'%(self.output.samples.shape))
                # If input and output are both defined, check for other necessary components.               
                if self.pushforward_dist is None:
                    print('WARNING: attribute `pushforward_dist` undefined. Necessary for `solve()`')
        
                if self.observed_dist is None:
                    print('WARNING: attribute `observed_dist` undefined. Necessary for `solve()`')
        
                if self.posterior_dist is None:
                    print('Posterior distribution is empty. Inverse Problem not yet solved.')
        
            else:
                print('You have yet to specify an output set. \
                        Please do so (either manually or with the `problem_set.mapper` module)')
 
        else:
            print('You have yet to specify an input set. \
                    Please generate a `sample_set` object and pass it to \
                    `problem_set` when instantiating the class.')
        pass

    def compute_pushforward_dist(self, method='sc', mirror=False, kwds=None):
        r"""
        TODO: Add this.
        """
        self.pf_pr_eval = None
        if method in ['sklearn', 'sk', 'k', 'skl', 'scikit', 'sckitlearn']:
            # Use sklearn's KDE module
            if kwds is not None:
                self.output.dist  = distributions.skde(self.output.samples, mirror, **kwds)
            else:
                self.output.dist  = distributions.skde(self.output.samples, mirror)
        else: # default behavior
            # Use Gaussian Kernel Density Estimation to estimate the density of the pushforward of the posterior
            self.output.dist  = distributions.gkde(self.output.samples) # attach gaussian_kde object to this handle.
            if mirror is True:
                print("Warning: You specified mirroring with the scipy gaussian kde. This is currently unsupported. Ignoring.")
        # Evaluate this using pset.pushforward_den.pdf()
        self.pushforward_dist = self.output.dist
        pass

    def set_observed_dist(self, dist=None, kwds=None, dim=None):
        r"""
        TODO: Add this.
        """
        # If `distribution = None`, we query the pushforward density for the top 5% to get a MAP estimate
        # TODO print warning about the aforementioned.
        # TODO check sizes, ensure dimension agreement
         
        
        if dist is not None:
            if (kwds is not None) and (dim is not None):
                self.observed_dist.set_dist(dim, dist, kwds)
            elif (kwds is None) and (dim is not None):
                self.observed_dist.set_dist(dim, dist)

            # Here we override the default errors printed by scipy.stats with our own.
            elif (kwds is None) and (distributions.supported_distributions(dist) is 'chi2' ):
                raise(AttributeError("If you are using a chi2 distribution, please pass `df` as a key-value pair in a dictionary. Ex: {'df': 20}."))
            elif (kwds is None) and (distributions.supported_distributions(dist) is 'beta'):
                raise(AttributeError("If you are using a Beta dist, please pass `a` and `b` as key-value pairs in a dictionary. Ex: {'a': 1, 'b': 1}"))
            # the following allows for manual overrides not using the parametric object.
            # e.g. kwds = {'loc': [1,2,3]}
            elif dim is None:
                logging.warn("OBS: No dimension specified. You will be using `scipy.stats` for your distributions instead of the parametric object. Be warned that functions like `.pdf` may not work as expected.")
                if kwds is not None:
                    self.observed_dist = distributions.assign_dist(dist, **kwds)
                else: 
                    self.observed_dist = distributions.assign_dist(dist)

        else:
            logging.warn("""No distribution specified. 
            Defaulting to normal around data_means with 0.5*data_standard_deviation""")
            loc = np.mean(self.output.samples, axis=0)
            scale = 0.5*np.std(self.output.samples, axis=0)
            self.observed_dist = distributions.assign_dist('normal', **{'loc':loc, 'scale':scale})
        pass
    
    def eval_pf_prior(self):
        r"""
        TODO
        """
        if self.pushforward_dist is None:
            raise(AttributeError("You are missing a defined pushforward distribution"))
        else:
            pf = self.pushforward_dist.pdf(self.output.samples).reshape(self.output.num_samples)
            self.pf_pr_eval = pf
        return pf
        
    def compute_ratio(self, samples):
        r"""
        Evaluates the ratio at a given set of samples 
        These samples should be the outputs of your map.
        
        :param sample_set: 
        :type sample_set: :class:`~/cbayes.sample.sample_set`
        
        :rtype: :class:`numpy.ndarray` of shape(num,)
        :returns: ratio of observed to pushforward density evaluations
        """
        n = samples.shape[0]
        try:
            obs = self.observed_dist.pdf(samples).prod(axis=1).reshape(n)
        except np.AxisError: # 1D case
            obs = self.observed_dist.pdf(samples).reshape(n)
        if len(samples) == len(self.output.samples):
            if np.allclose(samples.ravel(), self.output.samples.ravel()): # if you are asking for evaluation of the prior
                if self.pf_pr_eval is not None:
                    logging.warn("Detected stored evaluation. Reusing.")
                    pf = self.pf_pr_eval
                else: # if it hasn't been previously computed before, store it.
                    pf = self.eval_pf_prior()
            else: # if sizes dont match, we know it is a new set.
                pf = self.pushforward_dist.pdf(samples).reshape(n)
        else: # if you are asking for a different set of output samples to be evaluated,
            pf = self.pushforward_dist.pdf(samples).reshape(n)
        
        ratio = np.divide(obs, pf)
        ratio = ratio.ravel()
        return ratio
        
    def set_ratio(self):
        r"""
        TODO: rewrite description
        Runs compute_ratio and stores value in place.
        """
        data = self.output.samples
        ratio = self.compute_ratio(data)
        self.ratio = ratio
        pass
    
    def evaluate_posterior(self, samples):
        r"""
        TODO: rewrite description
        Evaluates but does not store an evaluation of the posterior.
        """
        return self.input.dist.pdf(samples)*self.compute_ratio(self.model(samples))
    
    
def save_sample_set():
    r"""
    TODO: Add this.
    """
    pass

def save_sample_set():
    r"""
    TODO: Add this.
    """
    pass
    
def save_problem_set():
    r"""
    TODO: Add this.
    """
    pass

def save_problem_set():
    r"""
    TODO: Add this.
    """
    pass

