1. Download and Install virtualenv (Option for no-sudo privilege user)
    Download virtualenv
        wget https://github.com/pypa/virtualenv/archive/15.1.0.tar.gz
    Untar virtualenv
        tar -xvzf 15.1.0.tar.gz
    Install virtualenv
        cd virtualenv-15.1.0
        python virtualenv.py ~/  (assume ~ is the location for virtual environment)
    Start virtualenv
        source ~/bin/activate

2. Download and Install Numpy
    Follow https://scipy.org/install.html   

2. Download and Install SimpleITK
    pip install SimpleITK

3. Download and Install CUDA (GPU is used for decompositions only)

4. Download and Install niftyreg
    Follow http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install
    In order to use the GPU, please set the CUDA_FLAG to on. (Currently all registration steps are using CPU only)

5. Download and Install pycuda
    Follow https://wiki.tiker.net/PyCuda/Installation/Linux
