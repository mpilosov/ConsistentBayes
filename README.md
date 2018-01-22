# Consistent Bayes Toolbox

This library is meant to provide a straightforward implementation of the Consistent Bayesian framework for solving Stochastic Inverse Problems.

--- 
## Installation Instructions

If you create a fresh Anaconda environment with   
`conda create -n py3b python=3.6 -y; source activate py3b`  
or simply using your existing python distribution (run `which python`),  
execute `python setup.py install` in the main ConsistentBayes directory to install all dependencies.

You can verify that everything is working as expected by executing the unit tests using the command `nosetests`.

**Please make sure to run `nosetests` before making any commits.**

---
## Getting Started
see the `examples/` directory for both script and jupyter-notebook files that walk you through how to solve stochastic inverse problems in the Consistent Bayesian framework, which was developed in tandem by:
- Dr. Troy Butler, CU Denver Dept. of Mathematics & Statistical Sciences
- Dr. Timothy Wildey, Sandia National Laboratories
- Dr. Scott Walsh, and
- Michael Pilosov, MS, CU Denver Dept. of Mathematics & Statistical Sciences


You can find general overview in the explanations presented in the jupyter notebooks.
** As of 1/21/18, these are not yet integrated with the new object-oriented
framework implemented in version 0.3.0 **

For more detail, please see [this draft on arxiv](https://arxiv.org/abs/1704.00680) _in review_


--- 

Author: Michael Pilosov  
Updated: 1/21/2018

---
This software was released under an MIT open-source license.  
Please see `LICENSE` for more information.