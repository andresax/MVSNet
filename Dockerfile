FROM nvidia/cuda:9.0-devel-ubuntu16.04


ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV NUM_CORES 10

# # # Update Ubuntu Software repository
RUN apt-get update --fix-missing 
RUN apt-get -y upgrade
RUN apt-get update
RUN apt install -y --no-install-recommends apt-utils

ARG PYTHON=python3
RUN apt-get install -y     ${PYTHON}     ${PYTHON}-pip
RUN pip3 --no-cache-dir install --upgrade     pip     setuptools
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt-get update
# RUN apt-get upgrade -y python3.6 python3-pip
# RUN pip3 --no-cache-dir install --upgrade     pip     setuptools
RUN apt-get install -y python-opencv
# RUN ln -s $(which python3.6) /usr/local/bin/python

RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends libcudnn7=7.0.5.15-1+cuda9.0

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir opencv-python 
# RUN pip install --no-cache-dir tqdm
# RUN pip install --no-cache-dir ipdb
# RUN pip install --no-cache-dir pudb
# RUN pip install --no-cache-dir scikit-learn
# RUN pip install --no-cache-dir scikit-image
# RUN pip install --no-cache-dir sklearn-extras
# RUN pip install --no-cache-dir progressbar2
# RUN pip install --no-cache-dir pyserial
# RUN pip install --no-cache-dir 'gast==0.2.2'

# RUN pip --no-cache-dir install \
#         Pillow \
#         h5py \
#         ipykernel \
#         jupyter \
#         matplotlib \
#         numpy \
#         pandas \
#         scipy \
#         sklearn \
#         && \
#     python -m ipykernel.kernelspec

# Install TensorFlow GPU version.
# RUN pip --no-cache-dir install tensorflow_gpu==1.5

RUN mkdir /.config
RUN chmod 777 /.config
RUN mkdir /.config/pudb
RUN chmod 777 /.config/pudb
COPY pudb.cfg /.config/pudb/pudb.cfg


CMD ["bash"]


WORKDIR /exp/mvsnet

