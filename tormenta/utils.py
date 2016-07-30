# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:27:26 2016

@author: Federico Barabas
"""

import os
from tkinter import Tk, filedialog


def getFilename(title, types, initialdir=None):
    try:
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title=title, filetypes=types,
                                              initialdir=initialdir)
        root.destroy()
        return filename
    except OSError:
        print("No file selected!")


def getFilenames(title, types=[], initialdir=None):
    try:
        root = Tk()
        root.withdraw()
        filenames = filedialog.askopenfilenames(title=title, filetypes=types,
                                                initialdir=initialdir)
        root.destroy()
        return root.tk.splitlist(filenames)
    except OSError:
        print("No files selected!")


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt
