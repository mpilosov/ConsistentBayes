# Copyright (C) 2018 Michael Pilosov

r"""
Consistent Bayesian formulation and solution for posing and solving stochastic inverse problems.

sample :mod:`cbayes.sample` provides data structures to store sets of samples and formulate inverse problems.

solve :mod:`cbayes.solve` provides various methods for solving the stochastic inverse problem, including accept/reject, MCMC (to add), and surrogate posteriors (to add).

postProcess :mod:`cbayes.postProcess` provides several plotting utilities, sorting tools, metrics, and other functionality once the posterior has been computed.

"""
__all__ = ['sample', 'solve', 'postProcess']

