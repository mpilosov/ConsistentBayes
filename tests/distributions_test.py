from nose import with_setup # optional
from cbayes import supported_distributions
from scipy.stats import _continuous_distns

def test_supp_dist_norm():
    print('========== testing supported_distributions for Normal ==========')
    assert type(supported_distributions('Normal')) is _continuous_distns.norm_gen
    assert type(supported_distributions('Gauss')) is _continuous_distns.norm_gen
    assert type(supported_distributions('Gaussian')) is _continuous_distns.norm_gen
    assert type(supported_distributions('Norm')) is _continuous_distns.norm_gen
    assert type(supported_distributions('N')) is _continuous_distns.norm_gen

def test_supp_dist_uni():
    print('========== testing supported_distributions for Uniform ==========')
    assert type(supported_distributions('Uniform')) is _continuous_distns.uniform_gen
    assert type(supported_distributions('U')) is _continuous_distns.uniform_gen
    assert type(supported_distributions('Uni')) is _continuous_distns.uniform_gen
