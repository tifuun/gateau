#!/bin/sh

docker run --rm -ti \
	-v ./dist:/dist:ro \
	gateau-ubuntu-20-04 \
	sh -c '. /venv/bin/activate; pip install /dist/*.whl ; python -m gateau'

