from nose.tools import assert_equals
from nose.tools import with_setup
from cbayes import sample_set
from cbayes import problem
from cbayes import supported_distributions
from scipy.stats import _distn_infrastructure
from os import error


class TestProblem:
 
    def setup(self):
        self.S = sample_set()  # instantiate the class
        self.S.setup() # just get the default setup options (x ~ U[0,1]) 

    def teardown(self):
        self.S = None # remove it from memory in preparation for the next test.

    @classmethod
    def setup_class(cls):
        print("\n=============== Testing `problem` class ===============\n")
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

