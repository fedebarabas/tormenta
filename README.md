# Tormenta
Measurement control and analysis for optical microscopy

### Prerequisites
 - [LJM Library](https://labjack.com/support/software/installers/ljm)
 - [LJM Library Python wrapper](https://labjack.com/support/software/examples/ljm/python)

### Installation

 - Ubuntu

Download and install [pygame](http://pygame.org/wiki/index). Then:

```
$ sudo apt-get install python3-pip python3-h5py git
$ sudo pip3 install git+https://github.com/fedebarabas/lantz tifffile pyqtgraph
$ git clone https://github.com/fedebarabas/Tormenta
```

 - Windows

     - Install [WinPython 3.4](https://sourceforge.net/projects/winpython/files/).
     - Clone [lantz repo](https://github.com/fedebarabas/lantz). Open WinPython Command Prompt, go to its directory and run
    ```
    $ python setup.py install
    ```
     - Download pygame from [http://www.lfd.uci.edu/~gohlke/pythonlibs/] for Python 3.4 to `WinPython-64bit-3.4.4.1\python-3.4.4.amd64\`.
     - Open WinPython Command Prompt and run:
    ```
    $ pip install pygame-1.9.2a0-cp34-none-win_amd64.whl tifffile 
    ```
     - Clone [Tormenta repo](https://github.com/fedebarabas/tormenta)

### Launch Tormenta

 - Go to its folder and run:

```
$ python -m tormenta
```