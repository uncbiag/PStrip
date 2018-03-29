FROM nvidia/cuda:9.0-devel-ubuntu16.04

MAINTAINER Xu Han xhs400@cs.unc.edu

RUN apt-get update; \ 
    apt-get -y upgrade; \
    apt-get install -y \
    apt-utils \
    python-pip \
    git \
    wget \
    g++ \
    cmake \
    doxygen \
    vim

RUN python -m pip install --upgrade pip
RUN pip install numpy scipy
RUN python -m pip install scikit-build
RUN python -m pip install SimpleITK

RUN apt-get install -y \
    zlib1g-dev \
    libpng-dev

RUN git clone https://git.code.sf.net/p/niftyreg/git niftyreg-git; \
    cd niftyreg-git; \
    mkdir build; \
    mkdir install; \
    cd build; \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/niftyreg-git/install ../; \
    make; \
    make install

ENV NIFTYREG_INSTALL /niftyreg-git/install
ENV PATH $PATH:$NIFTYREG_INSTALL/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$NIFTYREG_INSTALL/lib

RUN apt-get -y update; \
    apt-get install -y \
    libboost-all-dev \
    build-essential \
    python-dev \
    python-setuptools \
    libboost-python-dev \
    libboost-thread-dev



RUN wget https://pypi.python.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/pycuda-2017.1.1.tar.gz; \
    tar xfz pycuda-2017.1.1.tar.gz; \
    cd pycuda-2017.1.1; \
    python configure.py \
    --cuda-root=/usr/local/cuda \
    --cudadrv-lib-dir=/usr/lib/x86_64-linux-gnu \
    --boost-inc-dir=/usr/include \
    --boost-lib-dir=/usr/lib \
    --boost-python-libname=boost_python \
    --boost-thread-libname=boost_thread \
    --no-use-shipped-boost; \
    make install; \
    rm -rf pycuda-2017.1.1.tar.gz

RUN git clone https://github.com/lebedov/scikit-cuda.git; \
    cd scikit-cuda; \
    python setup.py install

RUN git clone https://github.com/uncbiag/PStrip.git

RUN apt-get install -y \
    zip \
    unzip

WORKDIR /PStrip
RUN wget https://github.com/uncbiag/PStrip/releases/download/v1.0.0-alpha/data.zip; \
    unzip data.zip; \
    rm data.zip

WORKDIR /
RUN mkdir input; \
    mkdir output


