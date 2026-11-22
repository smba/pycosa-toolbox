pycosa: configuration sampling toolbox
======================================

pycosa is a toolbox for sampling and analyzing configurations of
variability models: it models a system's configuration space as a
propositional formula over binary configuration options and provides
several sampling strategies (including exact uniform sampling), exact
configuration counting, and metrics to assess and compare sample quality.

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/smba/pycosa-toolbox.git@main

See the `README <https://github.com/smba/pycosa-toolbox#readme>`_ for a
usage guide.

API reference
-------------

Variability models
------------------

.. autosummary::
   :toctree: api
   :recursive:

   pycosa.modeling
   pycosa.parsing

Sampling
--------

.. autosummary::
   :toctree: api
   :recursive:

   pycosa.sampling
   pycosa.counting

Analysis
--------

.. autosummary::
   :toctree: api
   :recursive:

   pycosa.metrics
   pycosa.learning
   pycosa.experiments

Utilities
---------

.. autosummary::
   :toctree: api
   :recursive:

   pycosa.util
   pycosa.plotting
   pycosa.compatibility
   pycosa.lib

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
