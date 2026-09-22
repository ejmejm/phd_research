"""Connectivity, Credit Assignment, and the Speed of Learning.

Code for the multi-MNIST problem and the experiments in the paper, plus the
model and algorithm machinery for follow-up work on learned connectivity.

Layout:
    data.py         the multi-MNIST problem
    models/         PaddedMLP (the paper), BlockSparseMLP, DynamicNetwork
    algorithms/     connectivity algorithms: static, dense transition, SET, DEEP-R
    training.py     the training loop shared by all of them
    experiment.py   config -> run -> summary
    analysis/       plotting and run-export helpers used by the notebooks
"""

from .utils import register_resolvers

register_resolvers()

__version__ = '1.0.0'
