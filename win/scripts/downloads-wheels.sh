#!/bin/sh

pip download \
	-r win/requirements.txt \
	-d win/pipcache \
	--platform win_amd64 \
	--python-version 3.12 \
	--only-binary=:all:

pip download \
	pip-tools build twine \
	-d win/pipcache \
	--platform win_amd64 \
	--python-version 3.12 \
	--only-binary=:all:

