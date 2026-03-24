#!/bin/sh

docker run --rm -ti \
	-v .:/play \
	--name gateau-debian-bookworm \
	-v ./dist:/dist:ro \
	gateau-debian-bookworm \
	bash

