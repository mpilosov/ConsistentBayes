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
    def test_accept_reject(self):
        print('\n========== testing `solve.perform_accept_reject` ==========\n')

        def model(params): # dummlen(self.P.accept_inds)y model that generalizes to arbitrary dimensions
            #return np.multiply(2,data)
            return 2*params
            
        
        num_samples = 2000
        num_tests = 50
        
        for dim in range(1,6):
            err = []
            S = sample.sample_set()  # instantiate the class
            S.set_dim(dim) # just get the default setup options (x ~ U[0,1])
            for j in range(dim):
                S.set_dist('uniform', {'loc': 0, 'scale': 1}, j)
            S.generate_samples(num_samples)
            P = sample.map_samples_and_create_problem(S, model)
            P.compute_pushforward_dist()
            for j in range(dim):
                P.set_observed_dist('uniform', {'loc': 0.5, 'scale': 1}, j)
            P.set_ratio()
            
            for seed in range(num_tests):
                solve.problem(P, seed=seed)
                err.append(np.abs(len(P.accept_inds) - 
                                (0.5**P.input.dim)*P.input.num_samples))
            avg_missed = np.mean(err)
            print('dim %d: num of accepted inds out of %d trials in: %d'%(dim, num_tests, avg_missed))
            assert avg_missed < (0.05)*P.input.num_samples # want within 5% of num_samples
