from __future__ import division, with_statement, print_function

import Image
import numpy as np
import matplotlib.pyplot as plt

class Stack():
    
    def __init__(self, filename):

        self.image = Image.open(filename)
        self.size = self.image.size
        
        try:    
            while 1:
                self.image.seek(self.image.tell() + 1)
                
        except EOFError:
            pass
            
        self.nframes = self.image.tell()
        

    def frame(self, n):
        
        if n <= self.nframes:
            self.image.seek(n)
            data = np.array(self.image.getdata())
            
            return np.transpose(np.reshape(data, (self.size[0], self.size[1])))
        
        else:
            print(n, "is greater than the number of frames of the stack")
            

if __name__ == "__main__":
    
    data = Stack('muestra.tif')
    print(data.frame(2))
   
    print(data.nframes)