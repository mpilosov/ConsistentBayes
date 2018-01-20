#!/home/mpilosov/anaconda3/envs/py3/bin/python
import numpy as np
import scipy.stats as sstats
from scipy.stats import gaussian_kde as gkde

def supported_distributions(d):
    # currently supports 'normal' and 'uniform'
    # both take keyword arguments `loc` and `scale` of type `np.array` or `list`
    # method `sample_set.set_dist` just creates a handle for the chosen distribution. The longer of 
    # `loc` and `scale` is then inferred to be the dimension, which is written to sample_set.dim
    
    if d.lower() in ['gaussian', 'gauss', 'normal', 'norm', 'n']:
        d = 'normal'
    elif d.lower() in  ['uniform', 'uni', 'u']:
        d = 'uniform'

    D = {
        'normal': sstats.norm, 
        'uniform': sstats.uniform,
        }
    try:
        return D.get(d)
    except KeyError:
        print('Please specify a supported distribution. Type `?supported_distributions`')

class sample_set:
    def __init__(self):
        self.dim = None # dimension
        self.dist = None # the distribution on the space
        self.bounds = None # bounds on the space
        self.samples = None
        self.num_samples = None
        self.weights = None # weights for weighted KDE. If samples taken from dist, should be set to 1/N. #TODO default this
        self.seed = 0 # random number generator seed
 
        
    def set_dist(self, distribution, *kwags):
        # TODO describe how this is overloaded.
        # attach the scipy.stats._continuous_distns class to our sample set object
        if type(distribution) is str:
            distribution = supported_distributions(distribution)
        self.dist = distribution(*kwags)
   
    def set_dim(self, dimension):
        if dimension > 0:
            self.dim = int(dimension)
        else:
            os.error('Please specify an integer-valued dimension greater than zero.')
 
    def generate_samples(self, num_samples = 1E3):
        #TODO check if dimensions specified, if not, prompt user.
        # Since we want this function to work by default, we temporarily set a default. TODO remove this behavior.
        if self.dim is None: 
            print('dimension unspecified. Assuming 1D')
            self.dim = 1
        self.num_samples = num_samples
        self.samples = self.dist.rvs(size=(num_samples, self.dim))


class problem_set:
    def __init__(self, input_samples = None, output_samples = None):
        self.input = input_samples
        self.output = output_samples
        self.prior_dist = self.input.dist
        self.observed_dist = None
        self.pushforward_dist = None # kde object. should have rvs functionality. double check sizing.
        self.posterior_dist = None # this will be the dictionary object which we can use with .rvs(num_samples)
        self.accept_inds = None # indices into input_sample_set object associated with accepted samples from accept/reject
        self.ratio = None


    def get_problem_dims(self):
        if self.input.samples is not None:
            print('Your input space is %d-dimensional'%(self.input.dim))
            print('\t and is (%d, %d)'%(self.input.samples.shape))
        else:
            print('You have yet to specify an input set. Please generate a `sample_set` object and pass it to `problem_set` when instantiating the class.')
        if self.output.samples is not None:
            print('Your output space is %d-dimensional'%(self.output.dim))
            print('\t and is (%d, %d)'%(self.output.samples.shape))
        else:
            print('You have yet to specify an output set. Please do so (either manually or with the `problem_set.mapper` module)')
    def compute_pushforward_dist(self):
        # Use Gaussian Kernel Density Estimation to estimate the density of the pushforward of the posterior
        # Evaluate this using pset.pushforward_den.pdf()
        self.pushforward_dist = gkde(self.input.samples) # attach gaussian_kde object to this handle.

    def define_observed_dist(self, distribution_object):
        self.observed_dist = distribution_object    

    def compute_posterior_den(self):
        D = self.output.samples
        self.ratio = self.observed_dist.pdf(D) / self.pushforward_dist.pdf(D)

    def perform_accept_reject(self, normalize = True):
        # perform a standard accept/reject procedure by comparing normalized density values to u ~ Uniform[0,1]
        M = np.max(r)
        eta_r = self.ratio/M
        self.accept_inds = [i for i in range(num_samples) if eta_r[i] > np.random.rand() ] 

def mapper(sample_set, model):
    # pass a model, grab the input samples and map them to the data space.
    input_samples = sample_set
    output_samples = model(sample_set)
    pset = problem_set(input_samples, output_samples)
    return pset



