#!/bin/sh

qcow="$PWD/win/vm/6.gateau020.qcow2"
overlay="$PWD/win/tmp/overlay.qcow2"
qga_sock="win/tmp/qga.sock"
qmon_sock="win/tmp/qmon.sock"
qemu_pid_file="win/tmp/qemu.pid"
vfsd_sock="win/tmp/vfs.sock"

