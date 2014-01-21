# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:36:09 2013

@author: fbaraba
"""

import os

import numpy as np
import h5py as hdf


def store_stack(shape, dtype,
                data_name='image', filename=None, attributes=None):
    """Store binary data and measurement attributes in HDF5 format"""

    if filename is None:

        import tkFileDialog as filedialog
        from Tkinter import Tk

        root = Tk()
        filename = filedialog.askopenfilename(parent=root,
                                              title='Select file to pack')
        root.destroy()

    file_name = os.path.splitext(filename)

    # Data loading, reshaping, labelling
    data = np.memmap(filename, dtype=dtype, mode='r')
    data = data.reshape(shape)
    data = (data_name, data)

    attributes = [('nframes', shape[0]),
                  ('size', shape[1:3]),
                  ('nm_per_px', 133),
                  ('NA', 1.4),
                  ('lambda_em', 670)]

    store_file = hdf.File(file_name[0] + '.hdf5', "w")

    # data should be a list of tuples in the format ('data name', data)
    if type(data) is tuple:
        data = [data]

    for i in np.arange(len(data)):
        store_file.create_dataset(data[i][0], data=data[i][1])

    if attributes is None:
        print("No attributes were saved")

    else:
        for i in np.arange(len(attributes)):
            store_file.attrs[attributes[i][0]] = attributes[i][1]

    store_file.close()

if __name__ == "__main__":

    # shape = (nframes, width, height) not sure
#    shape = (110610, 85, 85)
#    dtype = np.dtype('>u2')     # big endian

#    # @home
#    shape = (1363, 185, 197)
#    dtype = np.dtype('<u2')      # little-endian

    # @MPI
    shape = (50000, 85, 85)
    dtype = np.dtype('>u2')     # big-endian

    store_stack(shape, dtype)
