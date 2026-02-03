#!/bin/sh

. ./win/scripts/lib.sh || {
	echo 'could not source lib.sh, correct PWD?'
	exit 1 
}

enforce_pwd ./win/scripts/build-in-vm.sh
load_paths
path_libexec
check_deps_qemu
find_iconv

check_empty_dist
check_vm_image
check_pipcache
make_tmpdirs
make_overlay

trap cleanup EXIT
trap cleanup HUP
trap cleanup INT

start_virtiofsd
start_qemu_with_overlay

wait_vm_start
wait_qga
mount_virtiofs

eecho "RUNNING BUILD SCRIPT IN GUEST"
#exec_out="$(qgasend '{"execute":"guest-exec","arguments":{"path": "Z:\\win\\bat\\build.bat","capture-output": true}}' )"
exec_out="$(qgasend '{"execute": "guest-exec", "arguments":{"path":"cmd.exe","arg":["/c","Z:\\win\\bat\\build.bat"], "capture-output": true}}' )"
exec_out_pid="$(echo "$exec_out" | jq -r '.return.pid')"

test -n "$exec_out_pid" || die "could not get build script pid on guest, bye!"
test ["$exec_out_pid" = "null"] && die "could not get build script pid on guest, bye!"

exec_did_exit=""

for i in $(seq 100)
do
	eecho "Waiting for build to complete (pid=$exec_out_pid, try=$i/100)..."
	exec_status=$(qgasend '{"execute": "guest-exec-status","arguments": {"pid": '"$exec_out_pid"'}}' )

	test -n "$exec_status" \
		|| die "could not get build script status on guest, bye!"

	descramble_qga_status "$exec_status"

	exec_exited=$(echo "$exec_status" | jq -r .return.exited)

	test -n "$exec_exited" \
		|| die "could not get build script status on guest, bye!"

	if [ "$exec_exited" = "true" ]
	then
		exitcode=$(echo "$exec_status" | jq -r .return.exitcode)
		exec_did_exit="yes"
		echo "Build completed with exit code: $exitcode"
		break
	fi
	sleep 1
done

test -n "$exec_did_exit" \
	|| die "Build script never exited on guest, bye!"

cleanup

eecho "Looks like the build succeeded, wheels should be under 'dist'."


