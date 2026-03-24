#!/bin/sh

docker build \
	--tag gateau-debian-bookworm-base \
	--file ./dockertest/Dockerfile.debian-bookworm-base \
	.

