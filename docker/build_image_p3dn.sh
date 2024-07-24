#!/bin/bash
DOCKER_BUILDKIT=1 nvidia-docker build -f Dockerfile --build-arg="INSTANCE_TYPE=p3dn" -t lancet:experiments .
