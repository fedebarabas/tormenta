from __future__ import division, with_statement, print_function

import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tkFileDialog as filedialog
from Tkinter import Tk

def openraw(filename, shape=None, datatype=np.dtype('uint16')):
    # 16-bit unsigned little-endian byte order

    fileobj = open(filename, 'rb')

    if shape==None:
        
        print('Shape not provided, loading it from inf file')
        rootname, ext = os.path.splitext(filename)
        inf_name = rootname + '.inf'
        inf_data = np.loadtxt(inf_name, dtype=str)
#       self.frame_rate = float(inf_data[22][inf_data[22].find('=') + 1:-1])
        n_frames = int(inf_data[29][inf_data[29].find('=') + 1:])
        frame_size = [int(inf_data[8][inf_data[8].find('=') + 1:]),
                      int(inf_data[8][inf_data[8].find('=') + 1:])]
        shape = (n_frames, frame_size[0], frame_size[1])
        print(shape)

    data = np.fromfile(fileobj, dtype=datatype).byteswap().reshape(shape)
    
    return data, shape

def show_frame(frame):
    # CCD plot
    plt.imshow(frame, cmap=plt.cm.jet, interpolation='nearest')
    plt.colorbar()
    plt.show()

class Stack():
    
    def __init__(self, filename=None):
        
        if filename==None:
            # File dialog
            root = Tk()
            filename = filedialog.askopenfilename(parent=root, 
                                                  initialdir=os.getcwd())
            root.destroy()

        self.filename = filename
        print(self.filename)
        os.chdir(os.path.split(self.filename)[0])
        
        rootname, self.ext = os.path.splitext(self.filename)
        
        if self.ext in ['.tiff', '.tif']:
            self.image = Image.open(self.filename)
            self.size = self.image.size
            
        else:
            print("Loading as raw image file")
            self.image, self.size = openraw(self.filename)

#        try:    
#            while 1:
#                self.image.seek(self.image.tell() + 1)
#                
#        except EOFError:
#            pass
#            
#        self.nframes = self.image.tell()
        
        
        

    def frame(self, n):
        
        if self.ext in ['.tiff', '.tif']:

            try:
                self.image.seek(n)
                data = np.array(self.image.getdata())
                
                return np.transpose(np.reshape(data, (self.size[0], self.size[1])))
            
            except EOFError:
                print(n, "is greater than the number of frames of the stack")
            
        else:
            return self.image[n]
            
            
if __name__ == "__main__":
    
#    %load_ext autoreload
#    %autoreload 2
    
    data = Stack()
#    print(data.frame(2))
    
#    show_frame(data.frame(20))
    
#    mean = np.array([data.frame(i).mean() for i in np.arange(10, 1000)])
#    print(mean.mean())
#    print(mean.std())
    
    
    
    