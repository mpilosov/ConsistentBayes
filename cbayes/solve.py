#!/home/mpilosov/anaconda3/envs/py3/bin/python
## Copyright 2018 Michael Pilosov

import numpy as np
import scipy.stats as sstats
from scipy.stats import gaussian_kde


def supported_distributions(d=None):
    # currently supports 'normal' and 'uniform'
    # both take kwags `loc` and `scale` of type `numpy.ndarray` or `list`
    # method `sample_set.set_dist` just creates a handle for the chosen distribution. The longer of 
    # `loc` and `scale` is then inferred to be the dimension, which is written to sample_set.dim

    # DICTIONARY OF SUPPORTED DISTRIBUTIONS:
    D = {
        'normal': sstats.norm, 
        'uniform': sstats.uniform,
        }

    if d is not None: 
        if d.lower() in ['gaussian', 'gauss', 'normal', 'norm', 'n']:
            d = 'normal'
        elif d.lower() in  ['uniform', 'uni', 'u']:
            d = 'uniform'

        try:
            return D.get(d)
        except KeyError:
            print('Please specify a supported distribution. Type `?supported_distributions`')
    else: # if d is unspecified, simply return the dictionary.
        return D


def assign_dist(distribution, *kwags):
    # TODO describe how this is overloaded.
    # If a string is passed, it will be matched against the options for `supported_distributions`
    # attach the scipy.stats._continuous_distns class to our sample set object
    if type(distribution) is str:
        distribution = supported_distributions(distribution)
    return distribution(*kwags)


### End of re-used methods.
class parametric_dist: # this is supposed to mimick scipy.stats
    def __init__(self, dim):
        self.dim = dim
        self.distributions = {str(d): None for d in range(dim)}

    def rvs(self, size = None):
        if size is None: # if nothing specified, just generate one draw from the distribution of the RV
            size = (self.dim, 1)
        #TODO parse dict, hcat results.
        pass 

    def args(self):
        pass

class gkde:
    # this is basically just a wrapper around `scipy.stats.gaussian_kde` that makes it conform to our syntax.
    def __init__(self, data):
        self.kde_object = gaussian_kde( data.transpose() )
        self.d = self.kde_object.d
        self.n = self.kde_object.n

    def rvs(self, size=1):
        if type(size) is tuple: 
            size=size[0]
        return self.kde_object.resample(size).transpose()
        #TODO write a test that makes sure this returns the correct shape
    
    def pdf(self, eval_points):
        p = self.kde_object.pdf( eval_points.transpose() ).reshape(eval_points.shape)
        #p = self.kde_object.pdf( eval_points.transpose() ) # alternative way to do the same thing
        #p = p[:,np.newaxis]
        return p
        #TODO write a test that makes sure this returns the correct shape
    


class sample_set:
    def __init__(self, size=(None, None)):
        # tuple `size` should be of format (num_samples, dim). 
        # Will write these attributes to class `sample_set`
        # If `size` is given as an integer, it is inferred to be dimension.
        if type(size) is tuple:
            self.num_samples = size[0] 
            self.dim = size[1] # dimension
        elif type(size) is int:
            self.dim = size
            self.num_samples = None # used as a default. 
            # will infer/set `num_samples` from call to `generate_samples`
        else:
            print('Please specify a valid size parameter. Defaulting to None.')
            self.dim = None
            self.num_samples = None
        
        self.dist = None # the distribution on the space. DEFAULT: unit normal in all dimensions.
        self.samples = None # this holds the actual samples we generate.
        self.seed = 0 # random number generator seed
        
        #self.bounds = None # bounds on the space
        #self.weights = None # weights for weighted KDE. 
        # If samples taken from dist, should be set to 1/N. #TODO default this

  
    def set_dim(self, dimension=1):
        if dimension > 0:
            self.dim = int(dimension)
        else:
            os.error('Please specify an integer-valued `dimension` greater than zero.')
    

    def set_num_samples(self, num_samples=1000):
        if num_samples > 0:
            self.num_samples = int(num_samples)
        else:
            os.error('Please specify an integer-valued `num_samples` greater than zero.')


    def set_dist(self, distribution='uniform', *kwags):
        self.dist = assign_dist(distribution, *kwags)

 
    def setup(self):
        # dummy function that runs the defaults to set up an unbounded 1D problem with gaussian prior.
        self.set_dim()
        self.set_num_samples()
        self.set_dist() 


    def generate_samples(self, num_samples=None, verbose=False):
        #TODO check if dimensions specified, if not, prompt user.
        # Since we want this function to work by default, we temporarily set a default. TODO remove this behavior.
        if self.dim is None:
            if verbose: 
                print('Dimension unspecified. Assuming 1D')
            self.dim = 1
        if num_samples is not None:
            if verbose:
                print("Number of samples declared, written to `sample_set.num_samples`.")
            self.num_samples = num_samples
        np.random.seed(self.seed) 
        self.samples = self.dist.rvs(size=(self.num_samples, self.dim))
        return self.samples

### End of `sample_set` class


class problem:
    def __init__(self, input_set=None, output_set=None, seed=None):
        self.input = input_set
        self.output = output_set
        self.prior_dist = self.input.dist
        self.pushforward_dist = self.output.dist # kde object. should have rvs functionality. TODO: double check sizing with test.
        self.posterior_dist = None # this will be the dictionary object which we can use with .rvs(num_samples)
        self.observed_dist = None
        self.accept_inds = None # indices into input_sample_set object associated with accepted samples from accept/reject
        self.ratio = None # the ratio is the posterior density evaluated on the `input_set.samples`
        if seed is None:
            self.seed = 0
        else:
            self.seed = seed


    def get_problem(self):
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
 

    def compute_pushforward_dist(self, method=None):
        # Use Gaussian Kernel Density Estimation to estimate the density of the pushforward of the posterior
        # Evaluate this using pset.pushforward_den.pdf()
        self.output.dist  = gkde(self.output.samples) # attach gaussian_kde object to this handle.
        self.pushforward_dist = self.output.dist


    def set_observed_dist(self, distribution=None, *kwags):
        # If `distribution = None`, we query the pushforward density for the top 5% to get a MAP estimate
        # TODO print warning about the aforementioned.
        # TODO check sizes, ensure dimension agreement
        if distribution is not None:
            self.observed_dist = assign_dist(distribution, *kwags)
        else:
            loc = np.mean(self.output.samples, axis=0)
            scale = 0.5*np.std(self.output.samples, axis=0)
            self.observed_dist = assign_dist('normal', loc, scale)


    def compute_ratio(self):
        data = self.output.samples
        self.ratio = self.observed_dist.pdf(data) / self.pushforward_dist.pdf(data)
        self.ratio = self.ratio.ravel()
       

    def perform_accept_reject(self, seed=None):
        # perform a standard accept/reject procedure by comparing normalized density values to u ~ Uniform[0,1]
        M = np.max(self.ratio)
        eta_r = self.ratio/M
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        self.accept_inds = [i for i in range(self.input.num_samples) if eta_r[i] > np.random.rand() ] 

### End of `problem_set` class


def map_samples_and_create_problem(input_sample_set, model):
    # pass a model, grab the input samples and map them to the data space.
    input_samples = input_sample_set.samples
    output_samples = model(input_samples) # make sure your model conforms to size (num_samples, dim)
    output_sample_set = sample_set(size=output_samples.shape)
    output_sample_set.samples = output_samples
    pset = problem(input_sample_set, output_sample_set)
    return pset



