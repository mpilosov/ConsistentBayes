## Copyright (C) 2018 Michael Pilosov

"""
This module contains unittests for :mod:`~cbayes.solve`
"""
import numpy as np
import cbayes.sample as sample
import cbayes.solve as solve
from nose import with_setup 


class TestSolve(object):
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

    @with_setup(setup, teardown) 
    def test_compute_ratio_and_accept_reject(self):
        print('\n========== testing `sample.problem_set.compute_ratio` \n \
            as well as `sample.problem_set.perform_accept_reject` ==========\n')
        # our testing model is q(x) = 2x, so we define a sample.problem_set where we expect half of the samples are accepted
        self.P.compute_pushforward_dist()
        self.P.set_observed_dist('uniform',0.5,1)
        self.P.set_ratio()
        err = []
        num_tests = 50
        for seed in range(num_tests):
            solve.problem(self.P, seed=seed)
            err.append(np.abs(len(self.P.accept_inds) - 0.5*self.P.input.num_samples))
        avg_missed = np.mean(err)
        print('mean num of accepted inds out of %d trials: %d'%(num_tests, avg_missed))
        assert avg_missed < 0.05*self.P.input.num_samples # want within 1% of num_samples
