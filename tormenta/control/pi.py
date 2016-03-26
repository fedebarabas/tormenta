# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:41:21 2014

@author: Federico Barabas
"""
# Taken from
# http://code.activestate.com/recipes/577231-discrete-pid-controller/
# The recipe gives simple implementation of a Discrete
# Proportional-Integral-Derivative (PID) controller. PID controller gives
# output value for error between desired reference input and measurement
# feedback to minimize error value.
# More information: http://en.wikipedia.org/wiki/PID_controller
#
# Example:
#
# p = PID(3.0, 0.4, 1.2)
# p.setPoint(5.0)
# while True:
#     pid = p.update(measurement_value)
#
#


class PI(object):
    """
    Discrete PI control
    """

    def __init__(self, setPoint, multiplier=1, kp=0, ki=0):

        self._kp = multiplier * kp
        self._ki = multiplier * ki
        self._setPoint = setPoint
        self.multiplier = multiplier

#        self._maxError = maxError
        self.error = 0.0
        self._started = False

    def update(self, currentValue):
        """
        Calculate PID output value for given reference input and feedback.
        I'm using the iterative formula to avoid integrative part building.
        ki, kp > 0
        """
        self.error = self.setPoint - currentValue

        if self.started:
            self.dError = self.error - self.lastError
            self.out = self.out + self.kp * self.dError + self.ki * self.error

        else:
            # This only runs in the first step
            self.out = self.kp * self.error
            self.started = True

        self.lastError = self.error

        return self.out

    def restart(self):
        self.started = False

    @property
    def started(self):
        return self._started

    @started.setter
    def started(self, value):
        self._started = value

    @property
    def setPoint(self):
        return self._setPoint

    @setPoint.setter
    def setPoint(self, value):
        self._setPoint = value

    @property
    def kp(self):
        return self._kp

    @kp.setter
    def kp(self, value):
        self._kp = value

    @property
    def ki(self):
        return self._ki

    @ki.setter
    def ki(self, value):
        self._ki = value
