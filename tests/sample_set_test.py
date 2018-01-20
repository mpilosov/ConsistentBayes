from nose.tools import assert_equals
from nose.tools import with_setup
from cbayes import sample_set
from cbayes import supported_distributions
from scipy.stats import _distn_infrastructure
from os import error


class TestSS:
 
    def setup(self):
        # print("TestSS:setup() before each test method")
        self.S = sample_set()  # instantiate the class

    def teardown(self):
        # print("TestSS:teardown() after each test method")
        self.S = None # remove it from memory in preparation for the next test.

    @classmethod
    def setup_class(cls):
        print("\n=============== Testing `sample_set` class ===============\n")
        pass
 
    @classmethod
    def teardown_class(cls):
        # print("teardown_class() after any methods in this class")
        pass

    @with_setup(setup, teardown) 
    def test_set_dim(self):
        print('\n========== testing `sample_set.set_dimension` setting ==========\n')
        self.S.set_dim()
        assert_equals(self.S.dim, 1) # check the default is correct.
        for i in range(1,6):
            self.S.set_dim(i)
            assert_equals(self.S.dim, i)


    @with_setup(setup, teardown) 
    def test_set_num_samples(self):
        print('\n========== testing `sample_set.set_num_samples` setting ==========\n')
        self.S.set_num_samples()
        assert_equals(self.S.num_samples, 100)
        for i in range(100,600,100):
            self.S.set_num_samples(i)
            assert_equals(self.S.num_samples, i)


    @with_setup(setup, teardown) 
    def test_set_dist(self):
        print('\n========== testing sample_set.set_dist` ==========\n')
        self.S.setup()
        D = supported_distributions() # dictionary of distributions
        # test `self.S.set_dist()` with no arguments.
        # print(type(self.S.dist))
        assert type(self.S.dist) is _distn_infrastructure.rv_frozen
        for dist_key in D.keys(): # the dist_keys are strings representing the distributions supported.
            self.S.set_dist(dist_key)
            assert type(self.S.dist) is type(D[dist_key]()) # check that it got instantiated to the expected type. 
            # now check that loc and scale were also set.
            for (l,s) in zip(range(0,5), range(1,6)):
                self.S.set_dist(dist_key, l, s)
                (loc, scale) = self.S.dist.args
                assert (loc == l and scale == s)

    @with_setup(setup, teardown) 
    def test_generate_samples(self):
        print('\n========== testing `sample_set.generate_samples` ==========\n')
        D = supported_distributions() # dictionary of distributions
        for n in [100,200,500]: # num_samples
            for d in [1,2,3]: # dim
                self.S.set_num_samples(n)
                self.S.set_dim(d)
                for dist_key in D.keys():
                    self.S.set_dist(dist_key)
                    self.S.generate_samples(self.S.num_samples)
                    assert self.S.samples.shape == (n, d) # ensure the samples were generated with the correct shape
