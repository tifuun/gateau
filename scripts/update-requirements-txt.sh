#!/bin/sh

set -e
rm -f requirements.txt
rm -f requirements.noextras.txt
pip-compile --all-build-deps --all-extras pyproject.toml
pip-compile --all-build-deps pyproject.toml --output-file requirements.noextras.txt

