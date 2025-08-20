#!/bin/sh

rsync --exclude docs --exclude .git --exclude src/gateau/libgateau.so -axv . mitakihara:gateau/gateau


