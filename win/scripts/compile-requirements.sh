#!/bin/sh

pip install pip-tools
pip-compile --all-build-deps pyproject.toml
mv requirements.txt win/requirements.txt


