# Tormenta
Measurement control and analysis for optical microscopy

### Installation

#### Optional prerequisites (not needed for offline testing)
Our setup uses LabJack's T7 as DAQ and a webcam as a position-sensitive detector. Don't install these libraries if you have different equipment or you just want to test the software without instruments (offline mode).
 - LabJack DAQ dependencies
     - [LJM Library](https://labjack.com/support/software/installers/ljm)
     - [LJM Library Python wrapper](https://labjack.com/support/software/examples/ljm/python)
 - Support for webcam acquisition
     - Pygame



#### Ubuntu
 - Run in terminal:

    ```
    $ sudo apt-get install python3-pip python3-h5py git
    $ sudo pip3 install comtypes lantz tifffile pyqtgraph
    $ git clone https://github.com/fedebarabas/Tormenta
    ```

#### Windows
- Install [WinPython 3.4](https://sourceforge.net/projects/winpython/files/).
- Browse to [Laboratory for Fluorescence Dynamics](http://www.lfd.uci.edu/~gohlke/pythonlibs/) and download tifffile for Python 3.4 to `$PATH\WinPython-64bit-3.4.4.1\python-3.4.4.amd64\`.
- Open WinPython Command Prompt and run:
    ```
    $ pip install comtypes lantz tifffile-2016.4.19-cp34-cp34m-win_amd64.whl
    ```
- Clone [Tormenta repo](https://github.com/fedebarabas/tormenta).

### Launch Tormenta
 - Open WinPython Command Prompt, go to tormenta's repository directory and run:

    ```
    $ python -m tormenta
    ```