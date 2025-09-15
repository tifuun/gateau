#!/bin/sh

rsync --delete --exclude docs --exclude .git --exclude src/gateau/libgateau.so -rvx . mitakihara:gateau/gateau


