.. Tormenta documentation master file, created by
   sphinx-quickstart on Tue Nov 22 15:13:01 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tormenta's documentation!
====================================

Tormenta is an open source software developed in our Lab to drive microscopy hardware. It is meant to be modular so it can be modified as desired. It currently consists in a Graphical User Interface along with several modules to control common microscopy hardware such as lasers, cameras, photon detectors and cameras.

Contents:

.. toctree::
   :maxdepth: 2

Initiation: main
=================
	
.. automodule:: tormenta
   :members:

The Graphical User Interface
============================

.. automodule:: control.control
   :members:

Instruments control
===================

Buffer for the instruments are created with these methods. They test if they are connected and if they aren't, they create dummy classes called Mockers to replace them for testing purposes.

.. automodule:: control.instruments
   :members:

   includeme

Contact
==================

federico.barabas[AT]cibion.conicet.gov.ar

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

