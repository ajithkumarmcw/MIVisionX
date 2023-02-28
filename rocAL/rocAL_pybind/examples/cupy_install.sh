#!/bin/bash

#Install Dependencies
mkdir workspace && cd workspace
apt-get update
apt install -y hipblas hipsparse rocrand rocthrust hipcub git g++ hipfft rocfft python3-dev

#Install CuPy
git clone https://github.com/ROCmSoftwarePlatform/cupy.git && cd cupy
export CUPY_INSTALL_USE_HIP=1
export HCC_AMDGPU_TARGET=gfx900
export ROCM_HOME=/opt/rocm
git submodule update --init
pip install -e . --no-cache-dir -vvvv