from __future__ import division, with_statement, print_function

import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tkFileDialog as filedialog
from Tkinter import Tk

class Stack():
    
    def __init__(self, filename=None):
        
        if filename==None:
            # File dialog
            root = Tk()
            filename = filedialog.askopenfilename(parent=root, 
                                                  initialdir=os.getcwd())
            root.destroy()

        self.filename = filename
        self.image = Image.open(filename)
        self.size = self.image.size
        
        try:    
            while 1:
                self.image.seek(self.image.tell() + 1)
                
        except EOFError:
            pass
            
        self.nframes = self.image.tell()
        
        print(self.filename)
        

    def frame(self, n):
        
        try:
            self.image.seek(n)
            data = np.array(self.image.getdata())
            
            return np.transpose(np.reshape(data, (self.size[0], self.size[1])))
        
        except EOFError:
            print(n, "is greater than the number of frames of the stack")
            

if __name__ == "__main__":
    
#    %load_ext autoreload
#    %autoreload 2
    
    data = Stack()
    print(data.frame(2))
   
    print(data.nframes)