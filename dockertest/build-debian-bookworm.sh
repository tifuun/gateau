#!/bin/sh

docker build \
	--tag gateau-debian-bookworm \
	--file ./dockertest/Dockerfile.debian-bookworm \
	.

