PROGRAM=$1
shift
OUTPUT=$1
shift

mkdir -p /models/nccl_perf && \
/opt/amazon/openmpi/bin/mpirun \
    --allow-run-as-root \
    --hostfile hostfile \
    -x FI_PROVIDER="efa" \
    -x FI_EFA_USE_DEVICE_RDMA=1 \
    -x RDMAV_FORK_SAFE=1 \
    -x NCCL_PXN_DISABLE=0 \
    -x LD_LIBRARY_PATH=/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/local/cuda/efa/lib:$LD_LIBRARY_PATH \
    --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    /root/nccl-tests/build/${PROGRAM} "$@" -g 1 > /models/nccl_perf/${OUTPUT}.log