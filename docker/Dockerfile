# syntax=docker/dockerfile:1.2
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG INSTANCE_TYPE
RUN test -n "$INSTANCE_TYPE" || (echo "INSTANCE_TYPE is not set" && false)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
  && apt-get update \
  && apt-get install -y python3-pip python3-dev git curl sudo wget ccache libz-dev \
                        libllvm-8-ocaml-dev libllvm8 llvm-8 llvm-8-dev             \
                        llvm-8-doc llvm-8-examples llvm-8-runtime                  \
                        clang-8 clang-tools-8 clang-8-doc libclang-common-8-dev    \
                        libclang-8-dev libclang1-8 clang-format-10                 \
                        python-clang-8 libfuzzer-8-dev lldb-8 lld-8                \
                        libc++-8-dev libc++abi-8-dev libomp-8-dev clang-tidy-8     \
                        openssh-server vim libtool autoconf libgflags-dev          \
                        libgoogle-glog-dev

# update glibc
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt install --only-upgrade libstdc++6 -y

RUN curl -s -L https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-x86_64.sh -o cmake.sh \
  && sh cmake.sh --skip-license --prefix=/usr/local/

ENV MOFED_VER 5.3-1.0.0.1
ENV OS_VER ubuntu20.04
ENV PLATFORM x86_64

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=aws
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH

# install aws efa software
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify

## Install NCCL
RUN git clone https://github.com/NVIDIA/nccl /opt/nccl \
    && cd /opt/nccl \
    && git checkout v2.12.12-1 \
    && make -j src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"


# install aws ofi plugin
RUN git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl \
    && cd /opt/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_VERSION} \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
       --with-libfabric=/opt/amazon/efa/ \
       --with-cuda=/usr/local/cuda \
       --with-nccl=/opt/nccl/build \
       --with-mpi=/opt/amazon/openmpi/ \
    && make && make install


RUN cd /usr/local/bin && ln -s /usr/bin/python3 python

# setup password-less SSH between nodes
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV HOME /root
ENV SSHDIR ${HOME}/.ssh/
RUN mkdir -p ${SSHDIR}

ADD ./resources/ssh/config ${SSHDIR}/config
ADD ./resources/ssh/id_rsa.mpi ${SSHDIR}/id_rsa
ADD ./resources/ssh/id_rsa.mpi.pub ${SSHDIR}/id_rsa.pub
ADD ./resources/ssh/id_rsa.mpi.pub ${SSHDIR}/authorized_keys

RUN chmod -R 600 ${SSHDIR}* && \
    chown -R ${USER}:${USER} ${SSHDIR}

RUN rm -fr ${HOME}/.openmpi && mkdir -p ${HOME}/.openmpi
RUN chown -R ${USER}:${USER} ${HOME}/.openmpi

# Python dependencies
RUN python3 -m pip install --upgrade pip
COPY ./resources/requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt
RUN python3 -m pip install black

# C++ dependencies
# ABSL
RUN git clone https://github.com/abseil/abseil-cpp.git /opt/abseil-cpp \
    && cd opt/abseil-cpp \
    && mkdir build && mkdir install && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/abseil-cpp/install -DCMAKE_CXX_STANDARD=14 .. \
    && cmake --build . --target install

# PROTOBUF
RUN cd ~ && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/protobuf-cpp-3.19.4.tar.gz \
    && cd /opt && tar -xzvf ~/protobuf-cpp-3.19.4.tar.gz \
    && rm ~/protobuf-cpp-3.19.4.tar.gz \
    && cd protobuf-3.19.4 \
    && ./autogen.sh \
    && ./configure \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# ORTOOLS
RUN cd ~ && wget https://github.com/google/or-tools/releases/download/v9.3/or-tools_amd64_ubuntu-20.04_v9.3.10497.tar.gz \
    && cd /opt && tar -xzvf ~/or-tools_amd64_ubuntu-20.04_v9.3.10497.tar.gz \
    && rm ~/or-tools_amd64_ubuntu-20.04_v9.3.10497.tar.gz \
    && cd or-tools_Ubuntu-20.04-64bit_v9.3.10497

# Clone and build Lancet and models
RUN git clone https://github.com/awslabs/accelerating-moe-training-via-whole-graph-comp-comm-overlapping --recursive /lancet \
    && cp -r /lancet/models /

ENV RAF_HOME=/lancet
ENV RAF_MODEL_HOME=/models
ENV PYTHONPATH=${RAF_HOME}/python/:${RAF_MODEL_HOME}/:${RAF_HOME}/3rdparty/tvm/python:$PYTHONPATH
ENV TVM_LIBRARY_PATH=$RAF_HOME/build/lib
ENV TVM_FFI=cython
ENV CUDA_HOME=/usr/local/cuda-11.3
ENV CUDNN_HOME=/usr
ENV NCCL_DIR=/opt/nccl/build
ENV ABSL_DIR=/opt/abseil-cpp/install

RUN cd /lancet && \
    ./scripts/src_codegen/run_all.sh
RUN mkdir /lancet/build
COPY ./resources/config_arch70.cmake /lancet
COPY ./resources/config_arch80.cmake /lancet
RUN if [ "$INSTANCE_TYPE" = "p3dn" ]; then \
        cp /lancet/config_arch70.cmake /lancet/config.cmake; \
    else \
        cp /lancet/config_arch80.cmake ./lancet/config.cmake; \
    fi
RUN cp /lancet/config.cmake /lancet/build/config.cmake
RUN cd /lancet/build && cmake .. && make -j$(($(nproc)-2))
RUN cd /lancet/3rdparty/tvm/ && make cython3

# Install Pytorch and model zoos
RUN TMPDIR=/var/tmp python3 -m pip install --cache-dir=$TMPDIR torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN TMPDIR=/var/tmp python3 -m pip install --cache-dir=$TMPDIR transformers==4.18.0
RUN TMPDIR=/var/tmp python3 -m pip install --cache-dir=$TMPDIR scikit-learn mxnet pytest pydot mpi4py ortools termcolor datasets
RUN DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed==0.5.8

# Install nccl-tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git $HOME/nccl-tests \
    && cd $HOME/nccl-tests \
    && make MPI=1 \
       MPI_HOME=/opt/amazon/openmpi/ \
       CUDA_HOME=/usr/local/cuda \
       NCCL_HOME=/opt/nccl/build \
       NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"

# install kubernetes
RUN TMPDIR=/var/tmp python3 -m pip install --cache-dir=$TMPDIR --ignore-installed kubernetes

# install tutel
RUN TMPDIR=/var/tmp python3 -m pip install --cache-dir=$TMPDIR --user --upgrade git+https://github.com/microsoft/tutel@v0.3.0

# fix numpy version to 1.23.5, otherwise TVM errors out
RUN python3 -m pip install numpy==1.23.5

RUN echo "$(echo -n 'source /env.sh\n'; cat ~/.bashrc)" > ~/.bashrc

RUN echo "\n\nHost *\nPort 6623\n" >> ~/.ssh/config
EXPOSE 6623

RUN if [ ! -f /usr/bin/bash ]; then \
        ln -s /bin/bash /usr/bin/bash; \
    fi

COPY ./resources/sharded_moe.py /usr/local/lib/python3.8/dist-packages/deepspeed/moe
COPY ./resources/env.sh .
ADD ./resources/experiment_scripts /models/experiment_scripts
RUN cd /models/experiment_scripts && mv ./* .. && cd .. && rmdir experiment_scripts
COPY ./resources/entrypoint /usr/local/bin
RUN chmod +x /models/*.sh
RUN chmod +x /usr/local/bin/entrypoint
ENTRYPOINT ["entrypoint"]