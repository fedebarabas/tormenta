# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:14:09 2015

@author: federico
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


class AnalysisWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        menuBar = QtGui.QMenuBar(self)
        fileMenu = menuBar.addMenu('&File')

        self.openStackAction = QtGui.QAction('Open stack...', self)
        self.openStackAction.setShortcut('Ctrl+O')
        self.openStackAction.setStatusTip('Open HDF5 stack file')
        self.openStackAction.triggered.connect(self.openStack)
        fileMenu.addAction(self.openStackAction)

        imv = pg.ImageView()
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(imv, 1, 0)

        grid.setRowMinimumHeight(0, 20)
        grid.setRowMinimumHeight(1, 600)
        grid.setColumnMinimumWidth(0, 600)

#    def openStack(self):
        


if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = AnalysisWidget()
    win.show()

    app.exec_()
