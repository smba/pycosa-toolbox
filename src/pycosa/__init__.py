"""
pycosa: a toolbox for sampling and analyzing configurations of
variability models.

The package provides tools to model the configuration space of a
software system (a propositional formula over binary configuration
options), to sample configurations from it with different strategies,
and to assess and compare the resulting samples:

- :mod:`pycosa.modeling` — propositional variability models
- :mod:`pycosa.parsing` — parsers for DIMACS and SPL Conqueror models
- :mod:`pycosa.sampling` — configuration sampling strategies
- :mod:`pycosa.counting` — exact model counting and uniform sampling
- :mod:`pycosa.metrics` — metrics for assessing sample quality
- :mod:`pycosa.learning` — identifying influential and interacting options
- :mod:`pycosa.util` — helpers for preparing measurement data
- :mod:`pycosa.plotting` — plotting utilities
- :mod:`pycosa.experiments` — synthetic ground-truth performance models
- :mod:`pycosa.compatibility` — readers for third-party data formats

Documentation: https://smba.github.io/pycosa-toolbox/
"""

__version__ = "0.2.0"
