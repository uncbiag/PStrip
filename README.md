# PStrip
This repository contains the code for the following paper (in writing)

## Setup
This code has been tested on Ubuntu 14.04 with Python 2.7 (Nvidia Titan X GPU, cuda 8.0)

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
**Download and Install niftyreg**
http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install
In order to use the GPU, please set the CUDA_FLAG to on. (Currently all registration steps are using CPU only)

**Download and Install pycuda**
Follow https://wiki.tiker.net/PyCuda/Installation/Linux
