# Consistent Bayes Toolbox

This library is meant to provide a straightforward implementation of the Consistent Bayesian framework for solving Stochastic Inverse Problems.

--- 
## Installation Instructions
Easiest way: execute `./fresh_install.sh`

This will create a new python kernel for you and link it to jupyter notebook.  
It will be called `test_py_env`. This file is a wrapper around the instructions that follow. You can activate this and have access to all the `cbayes` packages and dependencies with `source activate test_py_env`


If you create a fresh Anaconda environment with   
`conda create -n py3b python=3.6 -y; source activate py3b`  
or simply using your existing python distribution (run `which python`), you can 
execute `python setup.py install` in the main ConsistentBayes directory to install all dependencies.

You will have to link your kernel to jupyter if you want to use the notebooks.
Additionally, if you want to use widgets (interactive workbooks with sliders), you can check the series of commands listed in the file `fresh_install.sh` to ensure you have done so correctly.

You can verify that everything is working as expected by executing the unit tests using the command `nosetests`.

**Please make sure to run `nosetests` before making any commits.**

---
# Getting Started

It is suggested that you start with the `examples/CBayes.ipynb` file.
This notebook walks you through the method and several example files equipped with rich interactive multi-dimensional visualizations.  
A "bare-bones" python script that carries out the same computations can be found in `examples/consistentbayes_example.py`

See the `examples/` directory for both script and jupyter-notebook files that walk you through how to solve stochastic inverse problems in the Consistent Bayesian framework, which was developed in tandem by:
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