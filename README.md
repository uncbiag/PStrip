# PStrip
This repository contains the code for the following paper (in writing)

## Setup
This code has been tested on Ubuntu 14.04 with Python 2.7 (Nvidia Titan X GPU, cuda 8.0)

2017/10/19: CPU implementation is added. (No need for cuda, if only cpu is used)

2017/10/06: the code works on Ubuntu 16.04 with Python 2.7 (cuda 9.0)

**Download and Install virtualenv (Option for no-sudo privilege user)**
Download virtualenv
```bash
wget https://github.com/pypa/virtualenv/archive/15.1.0.tar.gz
```
Untar virtualenv
```bash
tar -xvzf 15.1.0.tar.gz
```
Install virtualenv
```bash
cd virtualenv-15.1.0
python virtualenv.py ~/  (assume ~ is the location for virtual environment)
```
Start virtualenv
```bash
source ~/bin/activate
```
**Download and Install Numpy**

Follow https://scipy.org/install.html

**Download and Install SimpleITK**
```bash
pip install SimpleITK
```
**Download and Install CUDA (GPU is used for decompositions only)**

CPU implementation of the decompostion is relatively slow, and we haven't uploaded it.

**Download and Install niftyreg**
http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install

For future purpose, n order to use the GPU, set the CUDA_FLAG to on. (Currently all registration steps are using CPU only)

2017/10/17: We add a 'remove_nan' function to replace the nan value to 0, after affine.   
2017/10/06: Newest niftyreg affine registration reg_aladin seems padding with NaN. It breaks the code at preprocessing.py.  

**Download and Install pycuda/scikit-cuda**
Follow https://wiki.tiker.net/PyCuda/Installation/Linux

obtain the scikit-cuda code at https://github.com/lebedov/scikit-cuda

2017/10/06: Do NOT use pip install for scikit-cuda, although it is instructed from scikit-cuda webside
```bash
DO NOT USE pip install scikit-cuda
```
I receive OSError: cusolver library not found, if installing from pip.
