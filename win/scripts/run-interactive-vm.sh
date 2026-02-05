#!/bin/sh

. ./win/scripts/lib.sh || {
	echo 'could not source lib.sh, correct PWD?'
	exit 1 
}

enforce_pwd ./win/scripts/run-interactive-vm.sh
load_paths
path_libexec
check_deps_qemu
check_kvm_group
find_iconv

make_tmpdirs
extract_zip_all
copy_pipcache
check_empty_dist
check_vm_image
check_pipcache
make_overlay

trap cleanup EXIT
trap cleanup HUP
trap cleanup INT

start_virtiofsd
#start_qemu_with_no_overlay
start_qemu_with_overlay

wait_vm_start
wait_qga
mount_virtiofs

wait_vm_finish_infinite




