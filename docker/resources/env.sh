#!/bin/bash
export RAF_HOME=/lancet
export RAF_MODEL_HOME=/models
export PYTHONPATH=${RAF_HOME}/python/:${RAF_MODEL_HOME}/:${RAF_HOME}/3rdparty/tvm/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$RAF_HOME/build/lib
export TVM_FFI=cython
export CUDA_HOME=/usr/local/cuda-11.3
export CUDNN_HOME=/usr
export NCCL_ROOT_DIR=/opt/nccl/build
export NCCL_DIR=/opt/nccl/build
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH=/opt/nvidia/nsight-systems/2022.2.1/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH
export MNM_SCH_FILE=/lancet/sch/latest.json
export RAF_SCH_FILE=/lancet/sch/latest.json
export ABSL_DIR=/opt/abseil-cpp/install
