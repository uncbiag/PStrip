# PStrip
This repository contains the code for the following paper (in writing)

We've add a pre-release version of the code, as well as the sample atlases and pca_basis brain images. You can copy the data folder into the PStrip folder.

We also have a dockerfile if you want to use docker. The dockerfile has all the setup commands for ubuntu16.04 with cuda 9.0

If you are not using docker, you may also need to update the niftyreg install location.   

Go to /PStrip/func_code, and change the nifty_bin in the niftyreg.conf file. The default location works for docker.  
  
Once done, you may go to PStrip/main_code and run 
```bash
python pstrip.py --help
```  
to view all the options.    
Or you may run   
```bash
python pstrip.py -i ../data/atlas/atlas_w_skull.nii  
```
which simply tries to extract the brain from atlas image.  

## Setup
This code has been tested on Ubuntu 14.04 with Python 2.7 (Nvidia Titan X GPU, cuda 8.0)

2017/10/19: CPU implementation is added. (No need for cuda, if only cpu is used)

2017/10/06: the code works on Ubuntu 16.04 with Python 2.7 (cuda 9.0)

This setup explains the procedure for PStrip. The exact command can be found in the Dockerfile.   

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

For future purpose, in order to use the GPU, set the CUDA_FLAG to on. (Currently all registration steps are using CPU only)

**Download and Install pycuda/scikit-cuda**
Follow https://wiki.tiker.net/PyCuda/Installation/Linux

obtain the scikit-cuda code at https://github.com/lebedov/scikit-cuda

2017/10/06: Do NOT use pip install for scikit-cuda, although it is instructed from scikit-cuda webside
```bash
DO NOT USE pip install scikit-cuda
```
I receive OSError: cusolver library not found, if installing from pip.
