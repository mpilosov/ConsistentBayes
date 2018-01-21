from nose.tools import assert_equals
from nose.tools import with_setup
from cbayes import sample_set
from cbayes import problem
#from cbayes import supported_distributions
#from scipy.stats import _distn_infrastructure
from cbayes import map_samples_and_create_problem
import numpy as np

class TestProblem:
 
    def setup(self):
        self.S = sample_set()  # instantiate the class
        self.S.setup() # just get the default setup options (x ~ U[0,1]) 
        self.S.generate_samples()
        def model(params): # dummlen(self.P.accept_inds)y model that generalizes to arbitrary dimensions
            #return np.multiply(2,data)
            return 2*params
        self.P = map_samples_and_create_problem(self.S, model)

    def teardown(self):
        self.S = None # remove it from memory in preparation for the next test.
        self.P = None

    @classmethod
    def setup_class(cls):
        print("\n=============== Testing `problem` class ===============\n")
        pass
 
    @classmethod
    def teardown_class(cls):
        # print("teardown_class() after any methods in this class")
        pass

    @with_setup(setup, teardown) 
    def test_compute_pushforward_dist(self):
        print('\n========== testing `problem.compute_pushforward_dist` setting ==========\n')
        # this test should just verify the dimensionality consistency
        # maybe run a test where you compare the data mean to a kde.rvs() (resampling) mean. 
        self.P.compute_pushforward_dist()
        assert_equals(self.P.pushforward_dist.d, self.P.output.dim)
        assert_equals(self.P.pushforward_dist.n, self.P.output.num_samples)
        
    @with_setup(setup, teardown) 
    def test_set_observed_dist(self):
        print('\n========== testing `problem.set_observed_dist` ==========\n')
        # not really a necessary test since it re-uses the assign_dist function.
        # however, we will still test its default behavior. 
        self.P.set_observed_dist()
        err = []
        num_tests = 50
        for seed in range(num_tests):
            np.random.seed(seed)
            n = 2000 # we want our sample mean from the parametric dist to be close to the mean of the data used to define it.
            #print(self.S.seed, self.P.seed)
            #print(np.mean(self.P.observed_dist.rvs(n), axis=0))
            #print(np.mean(self.P.output.samples, axis=0))
            err.append( np.linalg.norm(np.subtract( np.mean(self.P.observed_dist.rvs(n), axis=0),  
                                np.mean(self.P.output.samples, axis=0) ), ord=np.inf) )
        print('inf norm of errors over %d trials was %1.2e'%(num_tests, np.linalg.norm(err, np.inf)))
        assert np.linalg.norm(err, np.inf) < 2.5E-2


    @with_setup(setup, teardown) 
    def test_compute_ratio_and_accept_reject(self):
        print('\n========== testing `problem.compute_ratio` i\n \
            as well as `problem.perform_accept_reject` ==========\n')
        # our testing model is q(x) = 2x, so we define a problem where we expect half of the samples are accepted
        self.P.compute_pushforward_dist()
        self.P.set_observed_dist('uniform',0.5,1)
        self.P.compute_ratio()
        err = []
        num_tests = 50
        for seed in range(num_tests):
            self.P.perform_accept_reject(seed)
            err.append(np.abs(len(self.P.accept_inds) - 0.5*self.P.input.num_samples))
        avg_missed = np.mean(err)
        print('mean num of accepted inds out of %d trials: %d'%(num_tests, avg_missed))
        assert avg_missed < 0.05*self.P.input.num_samples # want within 1% of num_samples
