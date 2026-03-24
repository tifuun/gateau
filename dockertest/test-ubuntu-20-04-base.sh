#!/bin/sh

set -x

docker run --rm -ti \
	-v ./dist:/dist:ro \
	gateau-ubuntu-20-04-base \
	sh -c '. /venv/bin/activate; pip install /dist/*.tar.gz ; python -m gateau'

