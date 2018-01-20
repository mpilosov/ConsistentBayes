#!/home/mpilosov/anaconda3/envs/py3/bin/python
import numpy as np
import scipy.stats as sstats
from scipy.stats import gaussian_kde as gkde

def supported_distributions(d=None):
    # currently supports 'normal' and 'uniform'
    # both take keyword arguments `loc` and `scale` of type `np.array` or `list`
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
    

    def set_num_samples(self, num_samples=100):
        if num_samples > 0:
            self.num_samples = int(num_samples)
        else:
            os.error('Please specify an integer-valued `num_samples` greater than zero.')

 
    def set_dist(self, distribution='normal', *kwags):
        # TODO describe how this is overloaded.
        # If a string is passed, it will be matched against the options for `supported_distributions`
        # attach the scipy.stats._continuous_distns class to our sample set object
        if type(distribution) is str:
            distribution = supported_distributions(distribution)
        self.dist = distribution(*kwags)


    def setup(self):
        # dummy function that runs the defaults to set up an unbounded 1D problem with gaussian prior.
        self.set_dim()
        self.set_num_samples()
        self.set_dist() 


    def generate_samples(self, num_samples = None):
        #TODO check if dimensions specified, if not, prompt user.
        # Since we want this function to work by default, we temporarily set a default. TODO remove this behavior.
        if self.dim is None: 
            print('Dimension unspecified. Assuming 1D')
            self.dim = 1
        if num_samples is not None:
            print("Number of samples specified. Writing this value to `sample_set.num_samples`.")
            self.num_samples = num_samples
        
        self.samples = self.dist.rvs(size=(self.num_samples, self.dim))
        return self.samples

### End of `sample_set` class


class problem_set:
    def __init__(self, input_samples = None, output_samples = None):
        self.input = input_samples
        self.output = output_samples
        self.prior_dist = self.input.dist
        self.observed_dist = None
        # self.pushforward_dist should actually just be a pointer to output.dist
        #self.pushforward_dist = None # kde object. should have rvs functionality. double check sizing.
        self.posterior_dist = None # this will be the dictionary object which we can use with .rvs(num_samples)

        self.accept_inds = None # indices into input_sample_set object associated with accepted samples from accept/reject
        self.ratio = None


    def get_problem_dims(self):
        if self.input.samples is not None:
            print('Your input space is %d-dimensional'%(self.input.dim))
            print('\t and is (%d, %d)'%(self.input.samples.shape))
        else:
            print('You have yet to specify an input set. \
                    Please generate a `sample_set` object and pass it to \
                    `problem_set` when instantiating the class.')
        if self.output.samples is not None:
            print('Your output space is %d-dimensional'%(self.output.dim))
            print('\t and is (%d, %d)'%(self.output.samples.shape))
        else:
            print('You have yet to specify an output set. \
                    Please do so (either manually or with the `problem_set.mapper` module)')


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


### End of `problem_set` class


def mapper(sample_set, model):
    # pass a model, grab the input samples and map them to the data space.
    input_samples = sample_set
    output_samples = model(sample_set)
    pset = problem_set(input_samples, output_samples)
    return pset



