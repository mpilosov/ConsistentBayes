from nose.tools import assert_equals
from cbayes import sample_set
from cbayes import supported_distributions
from scipy.stats import _continuous_distns
from os import error

class TestSS:
 
    def setup(self):
        print("TestSS:setup() before each test method")
        self.S = sample_set()  # instantiate the class

    def teardown(self):
        print("TestSS:teardown() after each test method")
        self.S = None # remove it from memory in preparation for the next test.

    @classmethod
    def setup_class(cls):
        print("setup_class() before any methods in this class")
 
    @classmethod
    def teardown_class(cls):
        print("teardown_class() after any methods in this class")
 
    def test_set_dim(self):
        for i in range(1,5):
            self.S.set_dim(i)
            assert_equals(self.S.dim, i)
