## Copyright (C) 2018 Michael Pilosov

"""
This module contains unittests for :mod:`~cbayes.solve`
"""

import cbayes.sample as sample
import cbayes.solve as solve
from nose import with_setup 


def setup():
    S = sample.sample_set()  # instantiate the class
    S.setup() # just get the default setup options (x ~ U[0,1]) 
    S.generate_samples()
    def model(params): # dummlen(self.P.accept_inds)y model that generalizes to arbitrary dimensions
        #return np.multiply(2,data)
        return 2*params
    P = sample.map_samples_and_create_problem(S, model)

def teardown():
    S = None # remove it from memory in preparation for the next test.
    P = None

@with_setup(setup, teardown) 
def test_compute_ratio_and_accept_reject():
    print('\n========== testing `sample.problem_set.set_ratio`\n \
        as well as `sample.problem_set.perform_accept_reject` ==========\n')
    # our testing model is q(x) = 2x, so we define a sample.problem_set where we expect half of the samples are accepted
    P.compute_pushforward_dist()
    P.set_observed_dist('uniform',0.5,1)
    P.set_ratio()
    err = []
    num_tests = 50
    for seed in range(num_tests):
        solve.problem(P, seed=seed)
        err.append(np.abs(len(self.P.accept_inds) - 0.5*P.input.num_samples))
    avg_missed = np.mean(err)
    print('mean num of accepted inds out of %d trials: %d'%(num_tests, avg_missed))
    assert avg_missed < 0.05*P.input.num_samples # want within 1% of num_samples
