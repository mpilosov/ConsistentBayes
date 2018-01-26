from cbayes import sample
from cbayes import solve
import numpy as np

num_samples = 1000
num_tests = 50

def model(params): # dummlen(self.P.accept_inds)y model that generalizes to arbitrary dimensions
    #return np.multiply(2,data)
    return 2*params

for dim in range(1,6):
    err = []
    ones = np.ones(dim)
    S = sample.sample_set()  # instantiate the class
    S.set_dim(dim) # just get the default setup options (x ~ U[0,1])
    S.set_dist('uniform', 0*ones, 1*ones) 
    S.generate_samples(num_samples)
    P = sample.map_samples_and_create_problem(S, model)
    P.compute_pushforward_dist()
    P.set_observed_dist('uniform', 0.5*ones, 1*ones)
    P.set_ratio()
    
    for seed in range(num_tests):
        solve.problem(P, seed=seed)
        err.append(np.abs(len(P.accept_inds) - 
                        (0.5**P.input.dim)*P.input.num_samples))
    avg_missed = np.mean(err)
    print('dim %d: num of accepted inds out of %d trials in: %d'%(dim, num_tests, avg_missed))
    assert avg_missed < (0.05)*P.input.num_samples # want within 1% of num_samples
