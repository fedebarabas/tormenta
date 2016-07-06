# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:33:48 2016

@author: Federico Barabas
"""

import os
import numpy as np
from pyqtgraph import TableWidget


class FilterTable(TableWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.verticalHeader().hide()
        self.loadArray()
        self.resizeRowsToContents()
        self.itemChanged.connect(self.saveArray)
        self.dt = [('Posición', object), ('Antiposición', object),
                   ('Filtro', object), ('Tipo', object),
                   ('Fluoróforos', object)]

        self.cwd = os.getcwd()

    def defaultArray(self):

        f = [(1, 4, 'ZET642NF',    'Notch', ''),
             (2, 5, 'ET700/75m',   'Bandpass', 'Alexa647, Atto655'),
             (3, 6, 'FF01-725/40', 'Bandpass', 'Alexa700 (2 colores)'),
             (4, 1, '',            '',         ''),
             (5, 2, 'FF03-525/50', 'Bandpass', 'GFP'),
             (6, 3, '',            'Bandpass', '')]
        data = np.array(f, dtype=self.dt)
        return data

    def loadArray(self):
        try:
            filters = np.load(os.path.join(self.cwd, 'tormenta', 'control',
                                           'filter_array.npy'))
        except:
            filters = self.defaultArray()
        self.setData(filters)

    def saveArray(self):

        data = []
        rows = list(range(self.rowCount()))
        columns = list(range(self.columnCount()))
        for r in rows:
            row = []
            for c in columns:
                item = self.item(r, c)
                if item is not None:
                    row.append(str(item.value))
                else:
                    row.append(str(''))
            data.append(tuple(row))

        path = os.path.join(self.cwd, 'tormenta', 'control', 'filter_array')
        np.save(path, np.array(data, dtype=self.dt))
