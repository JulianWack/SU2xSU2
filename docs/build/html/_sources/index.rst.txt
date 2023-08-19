.. SU2xSU2 documentation master file, created by
   sphinx-quickstart on Fri Aug  4 13:19:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../../logo.png

This python package offers efficient simulation and data analysis routines for the
SU(2) x SU(2) Principal Chiral model. The key feature offered is the integration of 
Fourier Acceleration into the Hybrid Monte Carlo algorithm which leads to a 
significant reduction in the degree of critical slowing down.

Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project and some :ref:`examples <examples>`.

.. note::
   Currently the simulation is only supported for a two dimensional cubic lattice.

.. toctree::
   :caption: Contents:
   :maxdepth: 2

   usage
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`