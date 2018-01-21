# how the flow of problems works.

def model(params): # dummy model that generalizes to arbitrary dimensions
    return 2*params

input_sample_set = sample_set() # can pass it size=(num_samples,dim) to override defaults (1000,1)
input_sample_set.setup() # bunch of default options, 1D x ~ U[0,1]
input_sample_set.generate_samples()
problem = map_samples_and_create_problem(input_sample_set, model)
problem.compute_pushforward_dist() # no arguments, eventually will build support for `method` keyword
problem.set_observed_dist() # works like this, but also takes distributions and arguments. default puts normal with `loc` as data mean and with `scale` half the data std
problem.compute_ratio() # computes our ratio r.
problem.perform_accept_reject() # takes a random seed as an argument

# TODO:
# problem.compute_posterior() # multiplies prior times ratio.
# problem class needs set_seed module.

# BIG TODO: (almost there, little bit of code and then all the unit testing)
# finish parametric sampler class (in progress) that copies the
# syntax of `scipy.stats` distributions but can handle all sorts of crazy
# prior distributions (products of independent marginals of mixed types)
# so you can go param_dist.rvs(size=(40,3)) and it will iterate 
# through each dimension and its associated distribution, concatenating the results.
# This function will also be re-used by the custom parametric surrogate posterior, 
# so I am crafting it very carefully to be flexible.
