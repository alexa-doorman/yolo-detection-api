# Dockerfile with tensorflow gpu support on python3, opencv3.3
FROM tensorflow/tensorflow:1.3.0-py3

MAINTAINER MD Islam <mdislamwork@gmail.com>

# Based off of https://github.com/fbcotter/docker-tensorflow-opencv

RUN apt-get update

# Core linux dependencies. 
RUN apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev

# Python dependencies
RUN pip3 --no-cache-dir install \
    numpy \
    hdf5storage \
    h5py \
    scipy \
    py3nvml

WORKDIR /

RUN wget https://github.com/opencv/opencv/archive/3.3.0.zip \
	&& unzip 3.3.0.zip \
	&& mkdir /opencv-3.3.0/cmake_binary \
	&& cd /opencv-3.3.0/cmake_binary \
	&& cmake -DBUILD_TIFF=ON \
		  -DBUILD_opencv_java=OFF \
		  -DWITH_CUDA=OFF \
		  -DENABLE_AVX=ON \
		  -DWITH_OPENGL=ON \
		  -DWITH_OPENCL=ON \
		  -DWITH_IPP=ON \
		  -DWITH_TBB=ON \
		  -DWITH_EIGEN=ON \
		  -DWITH_V4L=ON \
		  -DBUILD_TESTS=OFF \
		  -DBUILD_PERF_TESTS=OFF \
		  -DCMAKE_BUILD_TYPE=RELEASE \
		  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
		  -DPYTHON_EXECUTABLE=$(which python3) \
		  -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
		  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
	&& make install \
	&& rm /3.3.0.zip \
	&& rm -r /opencv-3.3.0

RUN pip install cython \
    && pip install flask \
    && pip install flask-redis

RUN git clone https://github.com/thtrieu/darkflow.git \
    && cd darkflow \
    && python setup.py build_ext --inplace \
    && pip install . \
    && cd .. \
    && rm -rf darkflow

WORKDIR /src/app


CMD ["python", "api.py"]