# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:20:02 2015

@author: federico
"""
import os
import time
import pyqtgraph as pg
import numpy as np
import h5py as hdf
import tifffile as tiff

from PyQt4 import QtCore
from tkinter import Tk, filedialog


# taken from https://www.mrao.cam.ac.uk/~dag/CUBEHELIX/cubehelix.py
def cubehelix():
    return np.array(((0.00000, 0.00000, 0.00000),
                     (2.04510, 0.34680, 1.68810),
                     (4.02390, 0.71145, 3.45015),
                     (5.93385, 1.10160, 5.28360),
                     (7.76730, 1.51470, 7.18335),
                     (9.52935, 1.95585, 9.14685),
                     (11.20980, 2.42505, 11.16645),
                     (12.80865, 2.92485, 13.23960),
                     (14.32590, 3.46035, 15.36120),
                     (15.75645, 4.02900, 17.52615),
                     (17.10030, 4.63590, 19.73190),
                     (18.35490, 5.27850, 21.97080),
                     (19.52025, 5.96445, 24.23775),
                     (20.59635, 6.68865, 26.53020),
                     (21.57810, 7.45875, 28.84305),
                     (22.47060, 8.26965, 31.16865),
                     (23.26875, 9.12900, 33.50445),
                     (23.97510, 10.03170, 35.84280),
                     (24.59220, 10.98030, 38.18115),
                     (25.11750, 11.97735, 40.51185),
                     (25.55100, 13.02030, 42.83235),
                     (25.89780, 14.11425, 45.13755),
                     (26.15535, 15.25410, 47.41980),
                     (26.32620, 16.44240, 49.67655),
                     (26.41545, 17.67915, 51.90270),
                     (26.42055, 18.96180, 54.09060),
                     (26.34660, 20.29545, 56.24025),
                     (26.19615, 21.67500, 58.34655),
                     (25.96920, 23.10045, 60.40185),
                     (25.67085, 24.57180, 62.40360),
                     (25.30620, 26.08905, 64.34670),
                     (24.87525, 27.65220, 66.22860),
                     (24.38055, 29.25615, 68.04675),
                     (23.82975, 30.90090, 69.79350),
                     (23.22540, 32.58900, 71.46885),
                     (22.57005, 34.31535, 73.07025),
                     (21.86880, 36.07995, 74.59005),
                     (21.12420, 37.88025, 76.02825),
                     (20.34390, 39.71370, 77.38485),
                     (19.53045, 41.58030, 78.65220),
                     (18.68895, 43.47750, 79.83030),
                     (17.82195, 45.40530, 80.91660),
                     (16.93710, 47.35860, 81.91110),
                     (16.03695, 49.33485, 82.81125),
                     (15.12915, 51.33660, 83.61450),
                     (14.21625, 53.35620, 84.32085),
                     (13.30335, 55.39365, 84.93030),
                     (12.39555, 57.44640, 85.44030),
                     (11.50050, 59.51190, 85.85085),
                     (10.61820, 61.58760, 86.16450),
                     (9.75630, 63.67095, 86.37615),
                     (8.92245, 65.75940, 86.49090),
                     (8.11665, 67.85040, 86.50620),
                     (7.34655, 69.94140, 86.42460),
                     (6.61470, 72.03240, 86.24865),
                     (5.92875, 74.11575, 85.97580),
                     (5.29380, 76.19145, 85.60860),
                     (4.70985, 78.25695, 85.15215),
                     (4.18455, 80.31225, 84.60390),
                     (3.72300, 82.34970, 83.96895),
                     (3.32775, 84.36930, 83.24985),
                     (3.00135, 86.36595, 82.44660),
                     (2.75145, 88.34220, 81.56430),
                     (2.58060, 90.29295, 80.60550),
                     (2.49135, 92.21565, 79.57275),
                     (2.48625, 94.10520, 78.46860),
                     (2.57040, 95.96415, 77.30070),
                     (2.74890, 97.78995, 76.06650),
                     (3.01920, 99.57495, 74.77620),
                     (3.38895, 101.32425, 73.42980),
                     (3.85815, 103.03020, 72.03240),
                     (4.43190, 104.69280, 70.58910),
                     (5.10765, 106.30950, 69.10500),
                     (5.89050, 107.88030, 67.58265),
                     (6.78300, 109.40265, 66.02715),
                     (7.78260, 110.87655, 64.44360),
                     (8.89440, 112.29690, 62.83710),
                     (10.11840, 113.66370, 61.21275),
                     (11.45205, 114.97695, 59.57310),
                     (12.90300, 116.23410, 57.92835),
                     (14.46360, 117.43260, 56.27850),
                     (16.13895, 118.57755, 54.63120),
                     (17.92650, 119.66130, 52.99155),
                     (19.82880, 120.68640, 51.36465),
                     (21.84075, 121.65285, 49.75305),
                     (23.96490, 122.55810, 48.16695),
                     (26.19870, 123.40470, 46.60635),
                     (28.54215, 124.18755, 45.07890),
                     (30.99015, 124.91175, 43.58970),
                     (33.54525, 125.57475, 42.14385),
                     (36.20235, 126.17400, 40.74390),
                     (38.96145, 126.71715, 39.39750),
                     (41.81745, 127.19655, 38.10720),
                     (44.76780, 127.61985, 36.88065),
                     (47.81250, 127.98195, 35.72040),
                     (50.94645, 128.28795, 34.62900),
                     (54.16710, 128.53530, 33.61410),
                     (57.46935, 128.72655, 32.67825),
                     (60.85320, 128.86170, 31.82400),
                     (64.30845, 128.94330, 31.05900),
                     (67.83765, 128.97390, 30.38325),
                     (71.43315, 128.95095, 29.80185),
                     (75.09240, 128.87955, 29.31735),
                     (78.80775, 128.76225, 28.93485),
                     (82.57920, 128.59650, 28.65690),
                     (86.39655, 128.38740, 28.48350),
                     (90.25980, 128.13495, 28.42230),
                     (94.16385, 127.84425, 28.46820),
                     (98.10105, 127.51275, 28.63140),
                     (102.06885, 127.14810, 28.90935),
                     (106.05960, 126.74775, 29.30205),
                     (110.07075, 126.31680, 29.81460),
                     (114.09465, 125.85525, 30.44955),
                     (118.12620, 125.37075, 31.20435),
                     (122.16285, 124.85820, 32.07900),
                     (126.19695, 124.32780, 33.07605),
                     (130.22340, 123.77700, 34.19805),
                     (134.23710, 123.21090, 35.44245),
                     (138.23295, 122.62950, 36.80670),
                     (142.20585, 122.04045, 38.29590),
                     (146.15070, 121.44120, 39.90495),
                     (150.06240, 120.83685, 41.63640),
                     (153.93585, 120.22995, 43.48770),
                     (157.76340, 119.62560, 45.45630),
                     (161.54250, 119.02380, 47.53965),
                     (165.27060, 118.42710, 49.74030),
                     (168.93750, 117.84060, 52.05315),
                     (172.54320, 117.26430, 54.47565),
                     (176.08005, 116.70330, 57.00780),
                     (179.54550, 116.15760, 59.64450),
                     (182.93700, 115.63230, 62.38320),
                     (186.24435, 115.12995, 65.22390),
                     (189.47010, 114.65310, 68.15895),
                     (192.60660, 114.20175, 71.18580),
                     (195.65385, 113.78100, 74.30445),
                     (198.60420, 113.39340, 77.50725),
                     (201.45510, 113.04150, 80.79165),
                     (204.20655, 112.72530, 84.15510),
                     (206.85090, 112.44735, 87.58995),
                     (209.39070, 112.21275, 91.09365),
                     (211.82085, 112.02150, 94.66365),
                     (214.13880, 111.87615, 98.29230),
                     (216.34455, 111.77670, 101.97450),
                     (218.43300, 111.72570, 105.70770),
                     (220.40670, 111.72570, 109.48680),
                     (222.26055, 111.77925, 113.30415),
                     (223.99455, 111.88635, 117.15975),
                     (225.60870, 112.04955, 121.04340),
                     (227.10045, 112.26885, 124.95255),
                     (228.47235, 112.54680, 128.87955),
                     (229.72440, 112.88085, 132.82185),
                     (230.85150, 113.27610, 136.77435),
                     (231.86130, 113.73255, 140.72940),
                     (232.74870, 114.24765, 144.68445),
                     (233.51625, 114.82650, 148.63185),
                     (234.16395, 115.46655, 152.56650),
                     (234.69690, 116.17035, 156.48585),
                     (235.11255, 116.93535, 160.38225),
                     (235.41345, 117.76410, 164.25060),
                     (235.60215, 118.65405, 168.08580),
                     (235.68120, 119.60775, 171.88530),
                     (235.65315, 120.62265, 175.64400),
                     (235.51800, 121.69875, 179.35425),
                     (235.28340, 122.83605, 183.01350),
                     (234.94680, 124.03455, 186.61665),
                     (234.51585, 125.29425, 190.16115),
                     (233.99055, 126.61005, 193.63935),
                     (233.37600, 127.98450, 197.05125),
                     (232.67730, 129.41505, 200.38920),
                     (231.89445, 130.90170, 203.65065),
                     (231.03510, 132.44190, 206.83305),
                     (230.10180, 134.03310, 209.93130),
                     (229.09965, 135.67785, 212.94540),
                     (228.03120, 137.37105, 215.86770),
                     (226.90155, 139.11015, 218.70075),
                     (225.71835, 140.89515, 221.43690),
                     (224.48160, 142.72350, 224.07615),
                     (223.20150, 144.59520, 226.61595),
                     (221.87805, 146.50515, 229.05375),
                     (220.51635, 148.45335, 231.38955),
                     (219.12660, 150.43470, 233.61825),
                     (217.70880, 152.45175, 235.74240),
                     (216.26805, 154.49685, 237.76200),
                     (214.81200, 156.57255, 239.66940),
                     (213.34575, 158.67120, 241.47225),
                     (211.87185, 160.79280, 243.16290),
                     (210.39540, 162.93735, 244.74645),
                     (208.92660, 165.09720, 246.22290),
                     (207.46290, 167.27235, 247.58715),
                     (206.01705, 169.46280, 248.84685),
                     (204.58650, 171.66090, 249.99690),
                     (203.18145, 173.86920, 251.04240),
                     (201.80445, 176.08005, 251.98335),
                     (200.46060, 178.29345, 252.82230),
                     (199.15500, 180.50430, 253.55925),
                     (197.89020, 182.71515, 254.19420),
                     (196.67385, 184.91835, 254.73480),
                     (195.50595, 187.11390, 255.00000),
                     (194.39415, 189.29925, 255.00000),
                     (193.34100, 191.47185, 255.00000),
                     (192.34905, 193.62660, 255.00000),
                     (191.42595, 195.76605, 255.00000),
                     (190.56915, 197.88255, 255.00000),
                     (189.78885, 199.97610, 255.00000),
                     (189.08250, 202.04670, 255.00000),
                     (188.45265, 204.08925, 255.00000),
                     (187.90695, 206.10120, 255.00000),
                     (187.44540, 208.08255, 255.00000),
                     (187.06800, 210.03075, 254.69145),
                     (186.77985, 211.94580, 254.25285),
                     (186.58095, 213.82260, 253.76580),
                     (186.47385, 215.66115, 253.23795),
                     (186.46110, 217.45890, 252.66930),
                     (186.54015, 219.21840, 252.06750),
                     (186.71610, 220.93200, 251.43765),
                     (186.98385, 222.60480, 250.78485),
                     (187.35105, 224.22915, 250.11165),
                     (187.81260, 225.81015, 249.42315),
                     (188.36850, 227.34525, 248.72700),
                     (189.02130, 228.83190, 248.02575),
                     (189.76845, 230.27010, 247.32705),
                     (190.60995, 231.65985, 246.63090),
                     (191.54325, 233.00115, 245.94750),
                     (192.57090, 234.29400, 245.27940),
                     (193.68525, 235.53585, 244.63170),
                     (194.88885, 236.72925, 244.00695),
                     (196.17915, 237.87420, 243.41535),
                     (197.55105, 238.97325, 242.85435),
                     (199.00710, 240.02130, 242.33415),
                     (200.53965, 241.02090, 241.85730),
                     (202.14870, 241.97460, 241.42635),
                     (203.82915, 242.88495, 241.04640),
                     (205.57845, 243.74685, 240.72510),
                     (207.39660, 244.56540, 240.45990),
                     (209.27340, 245.34315, 240.26100),
                     (211.20885, 246.07755, 240.12585),
                     (213.20040, 246.77115, 240.06210),
                     (215.24040, 247.42650, 240.06975),
                     (217.32375, 248.04615, 240.15645),
                     (219.45045, 248.63010, 240.31965),
                     (221.61285, 249.18090, 240.56445),
                     (223.80840, 249.70110, 240.89340),
                     (226.02945, 250.19325, 241.30905),
                     (228.27090, 250.65480, 241.81140),
                     (230.53020, 251.09340, 242.40300),
                     (232.80225, 251.51160, 243.08385),
                     (235.07940, 251.90685, 243.85650),
                     (237.35655, 252.28680, 244.72350),
                     (239.63115, 252.64890, 245.68230),
                     (241.89810, 253.00080, 246.73545),
                     (244.14975, 253.34250, 247.88295),
                     (246.37845, 253.67655, 249.12225),
                     (248.58420, 254.00805, 250.45335),
                     (250.76190, 254.33700, 251.87880),
                     (252.90135, 254.66595, 253.39350),
                     (255.00000, 255.00000, 255.00000)),
                    dtype=np.ubyte)


# Check for same name conflict
def getUniqueName(name):

    n = 1
    while os.path.exists(name):
        if n > 1:
            name = name.replace('_{}.'.format(n - 1), '_{}.'.format(n))
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1

    return name


def attrsToTxt(filename, attrs):
    fp = open(filename + '.txt', 'w')
    fp.write('\n'.join('{}= {}'.format(x[0], x[1]) for x in attrs))
    fp.close()


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def fileSizeGB(shape):
    # self.nPixels() * self.nExpositions * 16 / (8 * 1024**3)
    return shape[0]*shape[1]*shape[2] / 2**29


def nFramesPerChunk(shape):
    return int(1.8 * 2**29 / (shape[1] * shape[2]))


def getFilenames(title, filetypes):
    try:
        root = Tk()
        filenames = filedialog.askopenfilenames(title=title,
                                                filetypes=filetypes)
        root.destroy()
        return root.tk.splitlist(filenames)
    except OSError:
        print("No files selected!")


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

        if self.filenames is None:
            self.filenames = getFilenames("Select HDF5 files",
                                          [('HDF5 files', '.hdf5')])

        else:
            self.filenames = [self.filenames]

        if len(self.filenames) > 0:
            for filename in self.filenames:

                file = hdf.File(filename, mode='r')

                for dataname in file:

                    data = file[dataname]
                    filesize = fileSizeGB(data.shape)
                    filename = (os.path.splitext(filename)[0] + '_' + dataname)
                    attrsToTxt(filename, [at for at in data.attrs.items()])

                    if filesize < 2:
                        time.sleep(5)
                        tiff.imsave(filename + '.tiff', data,
                                    description=dataname, software='Tormenta')
                    else:
                        n = nFramesPerChunk(data.shape)
                        i = 0
                        while i < filesize // 1.8:
                            suffix = '_part{}'.format(i)
                            partName = insertSuffix(filename, suffix, '.tiff')
                            tiff.imsave(partName, data[i*n:(i + 1)*n],
                                        description=dataname,
                                        software='Tormenta')
                            i += 1
                        if filesize % 2 > 0:
                            suffix = '_part{}'.format(i)
                            partName = insertSuffix(filename, suffix, '.tiff')
                            tiff.imsave(partName, data[i*n:],
                                        description=dataname,
                                        software='Tormenta')

                file.close()

        print(self.filenames, 'exported to TIFF')
        self.filenames = None
        self.thread.terminate()
        # for opening attributes this should work:
        # myprops = dict(line.strip().split('=') for line in
        #                open('/Path/filename.txt'))


class Grid():

    def __init__(self, viewBox, shape):

        self.showed = False
        self.vb = viewBox
        self.shape = shape

        self.yline1 = pg.InfiniteLine(pos=0.25*self.shape[0], pen='y')
        self.yline2 = pg.InfiniteLine(pos=0.50*self.shape[0], pen='y')
        self.yline3 = pg.InfiniteLine(pos=0.75*self.shape[0], pen='y')
        self.xline1 = pg.InfiniteLine(pos=0.25*self.shape[1], pen='y', angle=0)
        self.xline2 = pg.InfiniteLine(pos=0.50*self.shape[1], pen='y', angle=0)
        self.xline3 = pg.InfiniteLine(pos=0.75*self.shape[1], pen='y', angle=0)

    def update(self, shape):
        self.yline1.setPos(0.25*shape[0])
        self.yline2.setPos(0.50*shape[0])
        self.yline3.setPos(0.75*shape[0])
        self.xline1.setPos(0.25*shape[1])
        self.xline2.setPos(0.50*shape[1])
        self.xline3.setPos(0.75*shape[1])

    def toggle(self):

        if self.showed:
            self.hide()

        else:
            self.show()

    def show(self):
        self.vb.addItem(self.xline1)
        self.vb.addItem(self.xline2)
        self.vb.addItem(self.xline3)
        self.vb.addItem(self.yline1)
        self.vb.addItem(self.yline2)
        self.vb.addItem(self.yline3)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.xline1)
        self.vb.removeItem(self.xline2)
        self.vb.removeItem(self.xline3)
        self.vb.removeItem(self.yline1)
        self.vb.removeItem(self.yline2)
        self.vb.removeItem(self.yline3)
        self.showed = False


class Crosshair():

    def __init__(self, viewBox):

        self.showed = False

        self.vLine = pg.InfiniteLine(pos=0, angle=90, movable=False)
        self.hLine = pg.InfiniteLine(pos=0, angle=0,  movable=False)
        self.vb = viewBox

    def mouseMoved(self, evt):
        pos = evt
        if self.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def mouseClicked(self):
        try:
            self.vb.scene().sigMouseMoved.disconnect(self.mouseMoved)
        except:
            pass

    def toggle(self):
        if self.showed:
            self.hide()
        else:
            self.show()

    def show(self):
        self.vb.scene().sigMouseClicked.connect(self.mouseClicked)
        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
        self.vb.addItem(self.vLine, ignoreBounds=False)
        self.vb.addItem(self.hLine, ignoreBounds=False)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.vLine)
        self.vb.removeItem(self.hLine)
        self.showed = False


class ROI(pg.ROI):

    def __init__(self, shape, vb, pos, handlePos, handleCenter,
                 *args, **kwargs):

        self.mainShape = shape

        pg.ROI.__init__(self, pos, size=(128, 128), pen='y', *args, **kwargs)
        self.addScaleHandle(handlePos, handleCenter, lockAspect=True)
        vb.addItem(self)

        self.label = pg.TextItem()
        self.label.setPos(self.pos())
        self.label.setText('128x128')

        self.sigRegionChanged.connect(self.updateText)

        vb.addItem(self.label)

    def updateText(self):
        self.label.setPos(self.pos())
        size = np.round(self.size()).astype(np.int)
        self.label.setText('{}x{}'.format(size[0], size[1]))

    def hide(self, *args, **kwargs):
        super(ROI, self).hide(*args, **kwargs)
        self.label.hide()


class cropROI(pg.ROI):

    def __init__(self, shape, vb, *args, **kwargs):

        self.mainShape = shape

        pg.ROI.__init__(self, pos=(shape[0], shape[1]), size=(128, 128),
                        scaleSnap=True, translateSnap=True, movable=False,
                        pen='y', *args, **kwargs)
        self.addScaleHandle((0, 1), (1, 0))
