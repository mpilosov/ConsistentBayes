from cbayes import sample
from cbayes import solve
import numpy as np

num_samples = 1000
num_tests = 50

# Define your model. All it needs to do is return (n, d) as a size.
def model(params):  # model that generalizes to arbitrary dimensions
    # return np.multiply(2,data)
    return 2 * params


for dim in range(1, 6):
    err = []
    ones = np.ones(dim)
    # S = sample.sample_set()  # instantiate the object. Can also pass `size=(num_samples,dim)`
    # S.set_dim(dim) # set dimension
    # S.set_num_samples(num_samples) # set number of samples.
    # # # This is where the setup actually occurs # # #
    S = sample.sample_set(
        size=(num_samples, dim)
    )  # ... Alternatively, the three lines above
    S.set_dist(
        "uniform", {"loc": 0 * ones, "scale": 1 * ones}
    )  # uniform priors in all directions. 'normal' also available.
    S.generate_samples()  # generate samples, store them in the sample_set object.
    P = sample.map_samples_and_create_problem(
        S, model
    )  # map the samples, create new `sample_set` for outputs, put them together into a `problem_set` object.
    P.compute_pushforward_dist()  # gaussian_kde by default on the data space.
    P.set_observed_dist(
        "uniform", {"loc": 0.5 * ones, "scale": 1 * ones}
    )  # define your observed distribution.
    P.set_ratio()  # compute ratio (evaluate your observed and pushforward densities)
    # solve the problem several times to get an expectation.
    for seed in range(num_tests):
        solve.problem(P, seed=seed)  # default: perform accept/reject. `method='AR'`
        err.append(
            np.abs(len(P.accept_inds) - (0.5 ** P.input.dim) * P.input.num_samples)
        )
    avg_missed = np.mean(err)
    print(
        "dim %d: num of accepted inds out of %d trials in: %d"
        % (dim, num_tests, avg_missed)
    )
    assert avg_missed < (0.05) * P.input.num_samples  # want within 1% of num_samples
