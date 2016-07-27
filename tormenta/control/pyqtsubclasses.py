# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:50:36 2016

@author: Federico Barabas
"""

import os
import time
import numpy as np
import h5py as hdf
import tifffile as tiff
import multiprocessing as mp
from PyQt4 import QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree

import tormenta.control.guitools as guitools
import tormenta.analysis.registration as reg


class CamParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the camera during imaging
    """

    def __init__(self, andor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vpssTip = ("Faster vertical shift speeds allow for higher maximum \n"
                   "frame rates but reduce the pixel well depth, causing \n"
                   "degraded spatial resolution (smearing) for bright \n"
                   "signals. To improve the charge transfer efficiency the \n"
                   "vertical clock voltage amplitude can be increased at \n"
                   "the expense of a higher clock-induced charge.")

        preampTip = ("Andor recommend using the highest value setting for \n"
                     "most low-light applications")

        EMGainTip = ("A gain of x4-5 the read noise (see spec sheet) is \n"
                     "enough to render this noise source negligible. In \n"
                     "practice, this can always be achieved with EM Gain of \n"
                     "less than x300 (often much less). Pushing gain beyond \n"
                     "300 would give little or no extra SNR benefit and \n"
                     "would only reduce dynamic range.")

        hrrTip = ("Slower readout typically allows lower read noise and \n"
                  "higher available dynamic range, but at the expense of \n"
                  "slower frame rates")

        croppedTip = ("Ensure that no light is falling on the light \n"
                      "sensitive area outside of the defined region. Any \n"
                      "light collected outside the cropped area could \n"
                      "corrupt the images which were acquired in this mode.")

        loadMatrixTip = ("Load the affine matrix between channels for \n"
                         "online transformation while recording two-color \n"
                         "stacks")

        maxExp = andor.max_exposure.magnitude

        # Parameter tree for the camera configuration
        params = [{'name': 'Camera', 'type': 'str',
                   'value': andor.idn.split(',')[0]},
                  {'name': 'Field of view', 'type': 'group', 'children': [
                      {'name': 'Pixel size', 'type': 'float', 'value': 0.12,
                       'readonly': True, 'siPrefix': False, 'suffix': ' um'},
                      {'name': 'Shape', 'type': 'list',
                       'values': ['Full chip', '256x256', '128x128', '84x84',
                                  '64x64', 'Two-colors', 'Custom']},
                      {'name': 'Apply', 'type': 'action'},
                      {'name': 'Load matrix', 'type': 'action',
                       'tip': loadMatrixTip}
                  ]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                      {'name': 'Horizontal readout rate', 'type': 'list',
                       'values': andor.HRRates, 'tip': hrrTip},
                      {'name': 'Vertical pixel shift', 'type': 'group',
                       'children': [
                           {'name': 'Speed', 'type': 'list',
                            'values': andor.vertSpeeds[::-1], 'tip': vpssTip},
                           {'name': 'Clock voltage amplitude', 'tip': vpssTip,
                            'type': 'list', 'values': andor.vertAmps}]},
                      {'name': 'Frame Transfer Mode', 'type': 'bool',
                       'value': False},
                      {'name': 'Cropped sensor mode', 'type': 'group',
                       'children': [
                           {'name': 'Enable', 'type': 'bool', 'value': False,
                            'tip': croppedTip},
                           {'name': 'Apply', 'type': 'action'}]},
                      {'name': 'Set exposure time', 'type': 'float',
                       'value': 0.1, 'limits': (0, maxExp),
                       'siPrefix': True, 'suffix': 's'},
                      {'name': 'Real exposure time', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 's'},
                      {'name': 'Real accumulation time', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 's'},
                      {'name': 'Effective frame rate', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 'Hz'}]},
                  {'name': 'Gain', 'type': 'group', 'children': [
                      {'name': 'Pre-amp gain', 'type': 'list',
                       'values': list(andor.PreAmps), 'tip': preampTip},
                      {'name': 'EM gain', 'type': 'int', 'value': 1,
                       'limits': (0, andor.EM_gain_range[1]), 'tip': EMGainTip}
                      ]}]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)
        self._writable = True
        self.p.param('Camera').setWritable(False)

        self.fovGroup = self.p.param('Field of view')
        self.fovGroup.param('Load matrix').hide()
        self.fovGroup.param('Apply').hide()
        self.fovGroup.param('Shape').sigValueChanged.connect(self.shapeChanged)

        self.timeParams = self.p.param('Timings')
        self.cropModeParam = self.timeParams.param('Cropped sensor mode')
        self.cropModeParam.param('Apply').hide()
        self.cropModeEnableParam = self.cropModeParam.param('Enable')
        self.cropModeEnableParam.sigValueChanged.connect(self.cropButton)
        self.cropModeEnableParam.setWritable(False)
        self.frameTransferParam = self.timeParams.param('Frame Transfer Mode')
        self.frameTransferParam.sigValueChanged.connect(self.enableCropMode)

    def enableCropMode(self):
        value = self.frameTransferParam.value()
        if value:
            self.cropModeEnableParam.setWritable(True)
        else:
            self.cropModeEnableParam.setValue(False)
            self.cropModeEnableParam.setWritable(False)

    def cropButton(self):
        if self.cropModeEnableParam.value():
            self.cropModeParam.param('Apply').show()
        else:
            self.cropModeParam.param('Apply').hide()

    @property
    def writable(self):
        return self._writable

    @writable.setter
    def writable(self, value):
        self._writable = value
        self.p.param('Field of view').param('Shape').setWritable(value)
        self.timeParams.param('Frame Transfer Mode').setWritable(value)
        croppedParam = self.timeParams.param('Cropped sensor mode')
        croppedParam.param('Enable').setWritable(value)
        self.timeParams.param('Horizontal readout rate').setWritable(value)
        self.timeParams.param('Set exposure time').setWritable(value)
        vpsParams = self.timeParams.param('Vertical pixel shift')
        vpsParams.param('Speed').setWritable(value)
        vpsParams.param('Clock voltage amplitude').setWritable(value)
        gainParams = self.p.param('Gain')
        gainParams.param('Pre-amp gain').setWritable(value)
        gainParams.param('EM gain').setWritable(value)

    def attrs(self):
        attrs = []
        for ParName in self.p.getValues():
            Par = self.p.param(str(ParName))
            if not(Par.hasChildren()):
                attrs.append((str(ParName), Par.value()))
            else:
                for sParName in Par.getValues():
                    sPar = Par.param(str(sParName))
                    if sPar.type() != 'action':
                        if not(sPar.hasChildren()):
                            attrs.append((str(sParName), sPar.value()))
                        else:
                            for ssParName in sPar.getValues():
                                ssPar = sPar.param(str(ssParName))
                                attrs.append((str(ssParName), ssPar.value()))
        return attrs

    def cropModeOn(self):
        if self.cropModeEnableParam.value():
            self.cropModeParam.param('Apply').show()

    def shapeChanged(self):
        if self.fovGroup.param('Shape').value() == 'Two-colors':
            self.fovGroup.param('Apply').hide()
            self.fovGroup.param('Load matrix').show()
        elif self.fovGroup.param('Shape').value() == 'Custom':
            self.fovGroup.param('Apply').show()
            self.fovGroup.param('Load matrix').hide()
        else:
            self.fovGroup.param('Apply').hide()
            self.fovGroup.param('Load matrix').hide()


# HDF <--> Tiff converter
class TiffConverterThread(QtCore.QThread):

    def __init__(self, filename=None):
        super().__init__()

        self.converter = TiffConverter(filename, self)
        self.converter.moveToThread(self)
        self.started.connect(self.converter.run)
        self.start()


class TiffConverter(QtCore.QObject):

    def __init__(self, filenames, thread, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filenames = filenames
        self.thread = thread

    def run(self):

        self.filenames = guitools.getFilenames("Select HDF5 files",
                                               [('HDF5 files', '.hdf5')])

        if len(self.filenames) > 0:
            for filename in self.filenames:
                print('Exporting ', os.path.split(filename)[1])

                file = hdf.File(filename, mode='r')

                for dataname in file:

                    data = file[dataname]
                    filesize = guitools.fileSizeGB(data.shape)
                    filename = (os.path.splitext(filename)[0] + '_' + dataname)
                    attList = [at for at in data.attrs.items()]
                    guitools.attrsToTxt(filename, attList)

                    if filesize < 2:
                        time.sleep(5)
                        tiff.imsave(filename + '.tiff', data,
                                    description=dataname, software='Tormenta')
                    else:
                        n = guitools.nFramesPerChunk(data.shape)
                        i = 0
                        while i < filesize // 1.8:
                            suffix = '_part{}'.format(i)
                            partName = guitools.insertSuffix(filename, suffix,
                                                             '.tiff')
                            tiff.imsave(partName, data[i*n:(i + 1)*n],
                                        description=dataname,
                                        software='Tormenta')
                            i += 1
                        if filesize % 2 > 0:
                            suffix = '_part{}'.format(i)
                            partName = guitools.insertSuffix(filename, suffix,
                                                             '.tiff')
                            tiff.imsave(partName, data[i*n:],
                                        description=dataname,
                                        software='Tormenta')

                file.close()
                print('done')

        self.filenames = None
        self.thread.terminate()
        # for opening attributes this should work:
        # myprops = dict(line.strip().split('=') for line in
        #                open('/Path/filename.txt'))


class HtransformStack(QtCore.QObject):
    """ Transforms all frames of channel 1 using matrix H."""

    finished = QtCore.pyqtSignal()

    def run(self):
        Hname = guitools.getFilename("Select affine transformation matrix",
                                     [('npy files', '.npy')])
        H = np.load(Hname)
        xlim, ylim, cropShape = reg.get_affine_shapes(H)

        text = "Select files for affine transformation"
        filenames = guitools.getFilenames(text, types=[],
                                          initialdir=os.path.split(Hname)[0])
        for filename in filenames:
            print(time.strftime("%Y-%m-%d %H:%M:%S") +
                  ' Transforming stack ' + os.path.split(filename)[1])
            ext = os.path.splitext(filename)[1]
            filename2 = guitools.insertSuffix(filename, '_corrected')

            if ext == '.hdf5':
                with hdf.File(filename, 'r') as f0, \
                        hdf.File(filename2, 'w') as f1:

                    dat0 = f0['data']
                    dat1 = self.mpStack(dat0, xlim, ylim, H)

                    # Store
                    f1.create_dataset(name='data', data=dat1)
#                    f1['data'][:, -cropShape[0]:, :] = im1c

            elif ext in ['.tiff', '.tif']:
                with tiff.TiffFile(filename) as tt:

                    dat0 = tt.asarray()

                    if len(dat0.shape) > 2:

                        dat1 = self.mpStack(dat0, xlim, ylim, H)
                        tiff.imsave(filename2, dat1)

                    else:
                        tiff.imsave(guitools.insertSuffix(filename, '_ch0'),
                                    dat0[:128, :])
                        tiff.imsave(guitools.insertSuffix(filename, '_ch1'),
                                    reg.h_affine_transform(dat0[-128:, :], H))

            print(time.strftime("%Y-%m-%d %H:%M:%S") + ' done')

        self.finished.emit()

    def mpStack(self, dat0, xlim, ylim, H):

        # Multiprocessing
        n = len(dat0)
        cpus = mp.cpu_count()
        step = n // cpus
        chunks = [[i*step, (i + 1)*step] for i in np.arange(cpus)]
        chunks[-1][1] = n
        args = [[dat0[i:j, -128:, :], H] for i, j in chunks]
        pool = mp.Pool(processes=cpus)
        results = pool.map(transformChunk, args)
        pool.close()
        pool.join()
        im1c = np.concatenate(results[:])

        # Stack channels
        im1c = im1c[:, xlim[0]:xlim[1], ylim[0]:ylim[1]]
        im0 = dat0[:, :128, :][:, xlim[0]:xlim[1], ylim[0]:ylim[1]]
        return np.append(im0, im1c, 1)


def transformChunk(args):

    data, H = args

    n = len(data)
    out = np.zeros((n, 128, 266), dtype=np.uint16)
    for f in np.arange(n):
        out[f] = reg.h_affine_transform(data[f], H)

    return out
