"""
Generalized IPCA (GIPCA) Package

This package implements Generalized Instrumented PCA, which extends IPCA
by linking latent factors to observable macro variables.

Main Components:
- GeneralizedIPCA: Main estimation class
- Simulation tools: For Monte Carlo experiments
- Utility functions: Helper functions for data processing
"""

from .generalized_ipca import GeneralizedIPCA

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = ["GeneralizedIPCA"]
