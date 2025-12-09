#!/bin/sh

glibc_ver=$(ldd --version | grep 'GLIBC' | head -n 1 | rev | cut -d ' ' -f 1 | rev | tr '.' '_')

if [ -z "$glibc_ver" ]
then
	echo "Could not determine glibc ver, aborting."
	exit 2
fi

echo "found glibc $glibc_ver"
ldd --version
for f in dist/*
do
	renamed="$(echo "$f" | sed -e "s|any|manylinux_${glibc_ver}_x86_64|" -e "s|py3|cp39.cp310.cp311.cp312.cp313-none|")"
	if [ "$renamed" != "$f" ]
	then
		cp --reflink=auto "$f" "$renamed"
	fi
done

