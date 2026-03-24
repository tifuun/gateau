#!/bin/sh

docker build \
	--tag gateau-ubuntu-20-04-base \
	--file ./dockertest/Dockerfile.ubuntu-20-04-base \
	.

