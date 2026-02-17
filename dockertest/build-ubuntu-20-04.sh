#!/bin/sh

docker build \
	--tag gateau-ubuntu-20-04 \
	--file ./dockertest/Dockerfile.ubuntu-20-04 \
	.

