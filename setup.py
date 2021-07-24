#!/usr/bin/env python
# Copyright (C) 2018 Michael Pilosov

# Michael Pilosov 01/21/2018

"""
The python script for building the   
ConsistentBayes package and subpackages.
"""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="cbayes",
    version="0.4.1post1",
    description="Consistent Bayesian Inversion",
    author="Michael Pilosov",
    author_email="mpilosov@gmail.com",
    license="MIT",
    url="https://github.com/mpilosov/ConsistentBayes/",
    packages=["cbayes"],
    install_requires=["scipy", "numpy", "scikit-learn"],
)
