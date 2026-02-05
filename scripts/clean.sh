#!/bin/sh

rm -rf \
	*.cmake \
	.cmake \
	.coverage \
	.ruff_cache \
	.skbuild-info.json \
	CMakeCache.txt \
	CMakeFiles \
	CMakeInit.txt \
	Makefile \
	System\ Volume\ Information/ \
	\$RECYCLE.BIN/ \
	build \
	build.win \
	build.docker \
	build.dk \
	dist \
	docs \
	gateau_test \
	gsl \
	htmlcov/ \
	podman/ \
	requirements.txt.bak \
	src/gateau/libgateau.so \
	testfile \
	tmp \
	win/tmp \
	venv/ \
	wheel/ \
	.mesonpy* \

find . -name '__pycache__' -exec rm -rf {} \;
find . -name '*.pyc' -exec rm -rf {} \;
find . -name '*.pyo' -exec rm -rf {} \;
find win/dep -type d -maxdepth 1 -mindepth 1 -exec rm -rf {} \;


