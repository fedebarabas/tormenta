# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 17:39:46 2016

@author: Federico Barabas
"""

import time
import os

watchdir = '/home/lou/Documents/script/txts/'
contents = os.listdir(watchdir)
count = len(watchdir)
dirmtime = os.stat(watchdir).st_mtime

while True:
    newmtime = os.stat(watchdir).st_mtime
    if newmtime != dirmtime:
        dirmtime = newmtime
        newcontents = os.listdir(watchdir)
        added = set(newcontents).difference(contents)
        if added:
            print("Files added: %s" % (" ".join(added)))
        removed = set(contents).difference(newcontents)
        if removed:
            print("Files removed: %s" % (" ".join(removed)))

        contents = newcontents
    time.sleep(30)
