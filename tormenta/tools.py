# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:35:59 2016

@author: federico
"""
import os


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt
