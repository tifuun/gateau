#!/bin/sh

set -e

rm -f win/requirements.txt
pip install pip-tools
pip-compile \
	--all-extras \
	--all-build-deps \
	--output-file win/requirements.txt \
	pyproject.toml


