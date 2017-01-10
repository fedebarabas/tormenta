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
from PyQt4 import QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
import multiprocessing as mp
from lantz import Q_

import tormenta.control.guitools as guitools
import tormenta.utils as utils
from tormenta.analysis.stack import subtractChunk


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
                      {'name': 'Pixel size', 'type': 'float', 'value': 0.133,
                       'readonly': True, 'siPrefix': False, 'suffix': ' um'},
                      {'name': 'Shape', 'type': 'list',
                       'values': ['Full chip', '256x256', '128x128',
                                  'Two-colors 128px', '82x82',
                                  'Two-colors 82px', '64x64', 'Custom']},
                      {'name': 'Apply', 'type': 'action'},
                      {'name': 'View', 'type': 'list', 'values':
                       ['Single', 'Dual']},
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
        self.viewParam = self.fovGroup.param('View')
        self.viewParam.hide()
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
        if self.fovGroup.param('Shape').value().startswith('Two-colors'):
            self.fovGroup.param('Apply').hide()
            self.fovGroup.param('Load matrix').show()
            self.fovGroup.param('View').show()
        elif self.fovGroup.param('Shape').value() == 'Custom':
            self.fovGroup.param('Apply').show()
            self.fovGroup.param('Load matrix').hide()
            self.fovGroup.param('View').hide()
        else:
            self.fovGroup.param('Apply').hide()
            self.fovGroup.param('Load matrix').hide()
            self.fovGroup.param('View').hide()


class Calibrate3D(QtCore.QObject):

    sigDone = QtCore.pyqtSignal()

    def __init__(self, main, step=0.025, rangeUm=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = main
        self.step = Q_(step, 'um')
        self.rangeUm = Q_(rangeUm, 'um')

    def start(self):

        self.main.recWidget.writable = False
        self.main.tree.writable = False
        self.main.liveviewButton.setEnabled(False)

        path = self.main.recWidget.folderEdit.text()
        name = '3Dcalibration_step{}'.format(self.step)
        savename = guitools.getUniqueName(os.path.join(path, name) + '.tiff')

        steps = (self.rangeUm // self.step).magnitude
        self.main.focusWidget.zMove(-0.5*steps*self.step)
        self.main.focusWidget.zMove(self.step)
        time.sleep(0.1)

        stack = np.zeros((int(steps), self.main.shape[0], self.main.shape[1]),
                         dtype=np.uint16)

        for s in np.arange(steps, dtype=int):
            self.main.focusWidget.zMove(self.step)
            time.sleep(0.1)     # Waiting for the motor to get to new position
            image = self.main.img.image.astype(np.uint16)
            stack[s] = image

        tiff.imsave(savename, stack, software='Tormenta',
                    photometric='minisblack',
                    resolution=(1/self.main.umxpx, 1/self.main.umxpx),
                    extratags=[('resolution_unit', 'H', 1, 3, True)])

        self.main.focusWidget.zMove(-0.5*steps*self.step)

        self.main.recWidget.writable = True
        self.main.tree.writable = True
        self.main.liveviewButton.setEnabled(True)
        self.sigDone.emit()


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


class BkgSubtractor(QtCore.QObject):

    finished = QtCore.pyqtSignal()

    def __init__(self, main, window=101, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = main
        self.window = window

    def run(self):

        txt = "Select files for background sustraction"
        initialdir = self.main.recWidget.folderEdit.text()
        filenames = utils.getFilenames(txt, types=[], initialdir=initialdir)
        print(time.strftime("%Y-%m-%d %H:%M:%S") +
              ' Background subtraction started')
        for filename in filenames:
            print(time.strftime("%Y-%m-%d %H:%M:%S") +
                  ' Processing stack ' + os.path.split(filename)[1])
            ext = os.path.splitext(filename)[1]
            filename2 = utils.insertSuffix(filename, '_subtracted')
            if ext == '.hdf5':
                with hdf.File(filename, 'r') as f0, \
                        hdf.File(filename2, 'w') as f1:

                    self.data = f0['data'].value
                    if len(self.data) > self.window:
                        dataSub = self.mpSubtract()
                        f1.create_dataset(name='data', data=dataSub)
                    else:
                        print('Stack shorter than filter window --> ignore')

            elif ext in ['.tiff', '.tif']:
                with tiff.TiffFile(filename) as tt:

                    self.data = tt.asarray()
                    print(len(self.data), self.window)
                    if len(self.data) > self.window:
                        dataSub = self.mpSubtract()
                        tiff.imsave(filename2, dataSub)

                    else:
                        print('Stack shorter than filter window --> ignore')
        print(time.strftime("%Y-%m-%d %H:%M:%S") +
              ' Background subtraction finished')
        self.finished.emit()

    # Multiprocessing
    def mpSubtract(self):
        n = len(self.data)
        cpus = mp.cpu_count() - 1
        step = n // cpus
        chunks = [[i*step, (i + 1)*step] for i in np.arange(cpus)]
        chunks[-1][1] = n
        args = [self.data[i:j] for i, j in chunks]
        pool = mp.Pool(processes=cpus)
        results = pool.map(subtractChunk, args)
        pool.close()
        pool.join()
        return np.concatenate(results[:])
