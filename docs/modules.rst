Modules
***********************************

#1 -- illumination
=========================

The illumination control module is meant to control the wavelength, intensity and mode of illumination. It is able to drive three continuous wave lasers from different manufacturers. Each laser control is instantiated from the same Python class, so it is straightforward to change laser units or add a new one. Besides setting the output power of each one, there is a checkbox for setting the laser’s shutter state; i.e. switching on and off any of the illumination sources. The button ‘Get Intensities’ triggers the power measurement of each laser with a photodiode and reports the intensity at the sample using a previous calibration. 

This module also controls a motorized linear stage (Thorlabs APT Motor) that permits to switch between epi and total internal reflection illumination (TIRF). A routine was implemented to automatically find the optimum stage position that maximizes the intensity detected by the camera for TIRF.

.. automodule:: tormenta.control.lasercontrol
   :members:

#2 -- focus lock
=========================

The focus-lock module is designed to control a custom made focus stabilization unit. It monitors on a CMOS camera the reflection of a near-infrared laser beam sent to the sample through the objective in total internal reflection configuration. Any change in the sample-objective separation distance is detected as a shift in the position of the reflection on the CMOS sensor, and compensated by acting on the fine focus screw of the microscope with a stepper motor. 

A scrolling plot shows online the position (center of mass) of the reflected beam on the sensor. On the right there's a live output of the CMOS camera, which is especially useful during alignment of the near-infrared beam. From this module one can start the proportional and integrative feedback control system that locks the focus position, as well as modify the proportional and integrative constants.

.. automodule:: tormenta.control.focus
   :members:


#3 -- filter table
=========================

The emission filters module consists of an editable table containing information about the installed fluorescence emission filters, which eventually could be integrated with a control routine for a motorized filter wheel. 

.. automodule:: tormenta.control.filter_table
   :members:
