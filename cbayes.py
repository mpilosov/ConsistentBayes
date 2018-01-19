#!/home/mpilosov/anaconda3/envs/py3/bin/python
import numpy as np
import scipy.stats as sstats
from scipy.stats import gaussian_kde as gkde

class sample_set:
    def __init__(self):
        self.dim = 1 # dimension
        self.dist = None # the distribution on the space
        self.bounds = None # bounds on the space
        self.samples = None
        self.num_samples = None
        self.weights = None # weights for weighted KDE. If samples taken from dist, should be set to 1/N. #TODO default this
        self.seed = 0 # random number generator seed
 
    def set_dist(self, distribution_object):
        # attach the scipy.stats._continuous_distns class to our sample set object
        #TODO overload Type(String) to distribution_object
        self.dist = distribution_type
    
    def generate_samples(self, num_samples = 1E3):
        self.num_samples = num_samples

class problem_set:
    def __init__(self, input_samples = None, output_samples = None):
        self.input = input_samples
        self.output = output_samples
        self.prior_dist = self.input.dist
        self.observed_dist = None
        self.pushforward_dist = None
        #self.posterior_dist = None
        self.accept_inds = None # indices into input_sample_set object associated with accepted samples from accept/reject
        self.ratio = None

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



