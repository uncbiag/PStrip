# PStrip
This repository contains the code for the following paper (in writing)

## Setup
This code has been tested on Ubuntu 14.04 with Python 2.7 (Nvidia Titan X GPU, cuda 8.0)
2017/10/06: the code is being tested on Ubuntu 16.04 with Python 2.7 (cuda 9.0)

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

In order to use the GPU, please set the CUDA_FLAG to on. (Currently all registration steps are using CPU only)

2017/10/06: Newest niftyreg affine registration reg_aladin seems padding with NaN. It breaks the code at preprocessing.py. One way to work-around is to modify the following files under nifty-reg folder: 

path/to/niftyreg/reg-lib/_reg_aladin.cpp line 429
path/to/niftyreg/reg-lib/_reg_aladin_sym.cpp line 225
```bash
//  this->resamplingKernel->template castTo<ResampleImageKernel>()->calculate(interp, std::numeric_limits<T>::quiet_NaN());
  this->resamplingKernel->template castTo<ResampleImageKernel>()->calculate(interp, 0);
```


**Download and Install pycuda/scikit-cuda**
Follow https://wiki.tiker.net/PyCuda/Installation/Linux

obtain the scikit-cuda code at https://github.com/lebedov/scikit-cuda

2017/10/06: Do NOT use pip install for scikit-cuda, although it is instructed from scikit-cuda webside
```bash
DO NOT USE pip install scikit-cuda
```
I receive OSError: cusolver library not found, if installing from pip.
