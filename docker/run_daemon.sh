#!/bin/bash
DOCKER_BINARY="docker"
DEVICE_PREFIX=/dev/infiniband
${DOCKER_BINARY} run --pid=host -it -d --net=host --gpus all --privileged --cap-add=IPC_LOCK --shm-size=4g --ulimit memlock=-1:-1 --name lancet_exp lancet:experiments bash ${COMMAND[@]}
