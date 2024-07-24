#!/bin/bash
# requires two arguments: the test type and the arguments
source /env.sh
cd /models
TEST_TYPE=$1
shift

if [ $TEST_TYPE == "ds" -o $TEST_TYPE == "pt" -o $TEST_TYPE == "raf" ]; then
    if [ ${OMPI_COMM_WORLD_RANK} -eq 0 ]; then
        nsys profile -w true -t cuda,nvtx,cudnn,cublas -s none --capture-range=cudaProfilerApi --capture-range-end stop --cudabacktrace=true -x true -o /models/nsys_report python3 benchmark_${TEST_TYPE}.py "$@" 2>&1 | tee log_${OMPI_COMM_WORLD_RANK}.txt
    else
        python3 benchmark_${TEST_TYPE}.py "$@" 2>&1 | tee log_${OMPI_COMM_WORLD_RANK}.txt
    fi
else
    echo "Unknown test type: $TEST_TYPE"
    exit 1
fi