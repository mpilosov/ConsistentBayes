## Copyright (C) 2018 Michael Pilosov

"""
This module contains unittests for :mod:`~cbayes.solve`
"""

from nose import with_setup # optional
from cbayes import supported_distributions
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
# def test_supp_dist():
#     print('========== testing supported_distributions for no arguments ==========\n')
#     assert_equals(type(supported_distributions()), dict)
# 
# def test_supp_dist_norm():
#     print('========== testing supported_distributions for Normal ==========\n')
#     assert_equals(type(supported_distributions('Normal')), _continuous_distns.norm_gen)
#     assert_equals(type(supported_distributions('Gauss')), _continuous_distns.norm_gen)
#     assert_equals(type(supported_distributions('Gaussian')), _continuous_distns.norm_gen)
#     assert_equals(type(supported_distributions('Norm')), _continuous_distns.norm_gen)
#     assert_equals(type(supported_distributions('N')), _continuous_distns.norm_gen)
# 
# def test_supp_dist_uni():
#     print('========== testing supported_distributions for Uniform ==========\n')
#     assert_equals(type(supported_distributions('Uniform')), _continuous_distns.uniform_gen)
#     assert_equals(type(supported_distributions('U')), _continuous_distns.uniform_gen)
#     assert_equals(type(supported_distributions('Uni')), _continuous_distns.uniform_gen)
