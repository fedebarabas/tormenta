# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:03:01 2014

@author: fbaraba
"""


def does_not_overlap(p1, p2, min_distance):
    return max(abs(p1[1] - p2[1]), abs(p1[0] - p2[0])) > min_distance

def does_overlap(p1, p2, min_distance):
    return max(abs(p1[1] - p2[1]), abs(p1[0] - p2[0])) < min_distance