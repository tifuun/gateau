#!/bin/sh

set -x

docker run --rm -ti \
	-v ./dist:/dist:ro \
	gateau-ubuntu-20-04-base \
	bash

