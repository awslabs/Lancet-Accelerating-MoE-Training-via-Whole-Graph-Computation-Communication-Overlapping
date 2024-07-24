#!/bin/bash
# requires two arguments: the test type and the arguments
source /env.sh
cd /models
TEST_TYPE=$1
shift

if [ $TEST_TYPE == "ds" -o $TEST_TYPE == "pt" -o $TEST_TYPE == "raf" ]; then
    python3 benchmark_${TEST_TYPE}.py "$@" 2>&1 | tee log_${OMPI_COMM_WORLD_RANK}.txt
else
    echo "Unknown test type: $TEST_TYPE"
    exit 1
fi