## Copyright (C) 2018 Michael Pilosov

"""
This module contains unittests for :mod:`~cbayes.distributions`
"""

from nose import with_setup 
import cbayes.sample as sample
# import cbayes.solve as solve
import cbayes.distributions as distributions
from scipy.stats import _continuous_distns
from nose.tools import assert_equals


class TestDistributions(object):
    def setup(self):
        self.S = sample.sample_set()  # instantiate the class
        self.S.setup() # just get the default setup options (x ~ U[0,1]) 
        self.S.generate_samples()
        def model(params): # dummlen(self.P.accept_inds)y model that generalizes to arbitrary dimensions
            #return np.multiply(2,data)
            return 2*params
        self.P = sample.map_samples_and_create_problem(self.S, model)

    def teardown(self):
        self.S = None # remove it from memory in preparation for the next test.
        self.P = None

    def test_supp_dist_norm(self):
        print('========== testing distributions.supported_distributions for Normal ==========\n')
        assert_equals(type(distributions.supported_distributions('Normal')), _continuous_distns.norm_gen)
        assert_equals(type(distributions.supported_distributions('Gauss')), _continuous_distns.norm_gen)
        assert_equals(type(distributions.supported_distributions('Gaussian')), _continuous_distns.norm_gen)
        assert_equals(type(distributions.supported_distributions('Norm')), _continuous_distns.norm_gen)
        assert_equals(type(distributions.supported_distributions('N')), _continuous_distns.norm_gen)
    
    def test_supp_dist_uni(self):
        print('========== testing distributions.supported_distributions for Uniform ==========\n')
        assert_equals(type(distributions.supported_distributions('Uniform')), _continuous_distns.uniform_gen)
        assert_equals(type(distributions.supported_distributions('U')), _continuous_distns.uniform_gen)
        assert_equals(type(distributions.supported_distributions('Uni')), _continuous_distns.uniform_gen)
    
        