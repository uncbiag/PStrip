# PStrip
This repository contains the code for the following paper: (https://arxiv.org/abs/1711.05702)
```
@misc{Han2017Brain,
    title        = {Brain Extraction from Normal and Pathological Images: A Joint PCA/Image-Reconstruction Approach},
    author       = {X. Han and R. Kwitt and S. aylward and B. Menze and A. Asturias and P. Vespa and J. Van Horn and M. Niethammer},
    year         = {2017},
    howpublished = {arXiv:1711.05702}}
```
We support use of docker. If you want to use the GPU for decomposition, please follow below to install nvidia-docker: https://github.com/NVIDIA/nvidia-docker  

We also include a sample PCA basis images as well as atlases. You can download the zip file at (https://github.com/uncbiag/PStrip/releases/download/v1.0.0-alpha/data.zip). Unzip to the data folder. You can use your own datas, but you need to make sure the file name and the folder strucure are the same as mine.

## Usage [1,2,3 are for docker user; 4 is for non-docker user]
### 1. Use docker wrapper
The following script can be used, for example:
```
nvidia-docker run -i --rm=false -v=your_image_folder:/input:rw -t xhs400/pstrip:latest python /PStrip/main_code/pstrip.py -i /input/your_image_name
```
```
docker run -i --rm=false -v=your_image_folder:/input:rw -t xhs400/pstrip:latest python /PStrip/main_code/pstrip.py -i /input/your_image_name -p CPU
```
What the above script does are following:  
1. Pull the docker repository. (https://hub.docker.com/r/xhs400/pstrip/)  
2. Initiate a docker container. (Mount the image\_folder to '/input').
3. Run the python pstrip.py code: perform brain extraction on the input image and the results will be saved in the same folder.    

For all options for pstrip.py please go to main\_code folder, or you can run   
```
python pstrip.py -h
```
### 2. If you want to explore more, you may want to follow here: 
1. Pull the docker repository. (https://hub.docker.com/r/xhs400/pstrip/)
```
docker pull xhs400/pstrip
```
2. Run the docker container
```
nvidia-docker run -i --rm=false -v=your_image_folder:/input:rw pstrip:latest
```
Once log into the docker container, the source code is located in the '/PStrip'   

### 3. Use docker file: 
1. Download the Dockerfile in this repository or at (https://hub.docker.com/r/xhs400/pstrip/~/dockerfile/)  
2. Build the dockerfile.
3. Run the docker container

### 4. Build your own:
Even if you don't use docker, the Dockerfile is the best reference to build our code on your own.  
To test the source code, you may go to PStrip/main\_code and run the following code to view all the options
```bash
python pstrip.py --help
```  
Or you may run   
```bash
python pstrip.py -i ../data/atlas/atlas_w_skull.nii  
```
which simply tries to extract the brain from atlas image.  
For users without GPU, you can add '-p CPU'.



## Setup 
This section introduces the main steps to build PStrip. The exact command can be found in Dockerfile, which is the best referece.  
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

For future purpose, in order to use the GPU, set the CUDA_FLAG to on. (Currently all registration steps are using CPU only)

**Download and Install pycuda/scikit-cuda**
Follow https://wiki.tiker.net/PyCuda/Installation/Linux

obtain the scikit-cuda code at https://github.com/lebedov/scikit-cuda

2017/10/06: Do NOT use pip install for scikit-cuda, although it is instructed from scikit-cuda webside
```bash
DO NOT USE pip install scikit-cuda
```
I receive OSError: cusolver library not found, if installing from pip.
