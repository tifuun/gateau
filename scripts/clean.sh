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
	wheelhouse \
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
	venv/ \
	wheel/ \
	*.dll \
	vcpkg_installed \
	.mesonpy* \

find . -name '__pycache__' -exec rm -rf {} \;
find . -name '*.pyc' -exec rm -rf {} \;
find . -name '*.pyo' -exec rm -rf {} \;


