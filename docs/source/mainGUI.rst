Tormenta
********

Tormenta is provided with its own GUI to make it user-friendly. The global layout is defined in the widget called TormentaGUI, and individual layouts are defined within the code of each widget.

The communication with the different instruments is initialized as soon as Tormenta is started; it tests if they are connected and creates an object to control them. Otherwise, it creates a Mocker, which has the same attributes but doesn't correspond to any physical device, so it can run without hardware.

Graphical User Interface
========================

.. automodule:: tormenta.control.control
   :members:

Instruments control
====================

These methods create a buffer for the instruments. They test if they are connected and if they aren't, they create dummy classes called Mockers to replace them for testing purposes.

.. automodule:: tormenta.control.instruments
   :members:
