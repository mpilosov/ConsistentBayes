#!/bin/bash
source deactivate
conda remove -n test_py_env --all -y

conda create -n test_py_env python=3.6 -y && 
source activate test_py_env && python setup.py install && nosetests

python -m ipykernel install --name test_py_env --user
jupyter nbextension install --py --user widgetsnbextension
jupyter nbextension enable --py widgetsnbextension

