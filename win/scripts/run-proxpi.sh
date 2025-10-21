#!/bin/sh

if [ "$(realpath "$0" 2>/dev/null)" != "$(realpath "./win/scripts/run-proxpi.sh" 2>/dev/null)" ]
then
	echo "Must run from root of repo!!"
	exit 9
fi

PROXPI_CACHE_DIR=./win/proxpi flask --app proxpi.server run --host 0.0.0.0

