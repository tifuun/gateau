#!/bin/sh

python -m pip install -e . \
	--config-setting=setup-args=-Dgsl_static=enabled \
	--config-setting=setup-args=-Dhdf5_static=enabled

