#!/bin/sh

rsync \
	--delete \
	--exclude docs \
	--exclude .git \
	--exclude gsl \
	--exclude gsl.bak \
	--exclude src/gateau/libgateau.so \
	-rvx \
	. mitakihara:gateau/gateau


