#!/bin/sh

docker run --rm -ti \
	-v ./dist:/dist:ro \
	gateau-debian-bookworm \
	sh -c '. /venv/bin/activate; pip install /dist/*.whl ; python -m gateau'

