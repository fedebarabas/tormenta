# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 10:41:21 2014

@author: federico
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


class PI:
    """
    Discrete PI control
    """

    def __init__(self, P=0.0, I=0.0):

        self.Kp = P
        self.Ki = I

        self.setPoint = 0.0
        self.error = 0.0

    # TODO: check this
    def update(self, current_value):
        """
        Calculate PID output value for given reference input and feedback
        """

        self.error = self.set_point - current_value

        self.P_value = self.Kp * self.error

        PID = self.P_value + self.I_value + self.D_value

        return PID

    @property
    def setPoint(self):
        return self.setPoint

    @setPoint.setter
    def setPoint(self, value):
        self.setPoint = value

    @property
    def kp(self):
        return self.kp

    @kp.setter
    def kp(self, value):
        self.kp = value

    @property
    def ki(self):
        return self.ki

    @ki.setter
    def ki(self, value):
        self.ki = value
