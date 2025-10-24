#!/bin/sh

qcow="$PWD/win/vm/5.final.qcow2"
overlay="$PWD/win/tmp/overlay.qcow2"
qga_sock="win/tmp/qga.sock"
qmon_sock="win/tmp/qmon.sock"
qemu_pid_file="win/tmp/qemu.pid"
vfsd_sock="win/tmp/vfs.sock"

# echo to stderr
eecho() {
	echo "$@" >&2
}

# echo to stderr and exit with error
die() {
	eecho "$@"
	exit 1
}

cleanup() {
	eecho "CLEANING UP"

	if [ -n "$GATEAU_DEBUG" ]
	then
		eecho "GATEAU_DEBUG is set, pausing. Press enter to exit."
		read -r foo
		echo "$foo"
	fi

	eecho "STOPPING VM"
	qemu_pid="$(cat "$qemu_pid_file")"
	if [ -n "$qemu_pid" ]
	then
		for i in $(seq 20)
		do
			if ! kill -0 "$qemu_pid"
			then
				eecho "VM STOPPED (OR NEVER STARTED)!"
				break
			fi
			eecho "ASKING VM TO STOP (try $i/20)..."
			echo "system_powerdown" | \
				socat - UNIX-CONNECT:"$qmon_sock"
			sleep 5
		done

		while kill -0 "$qemu_pid"
		do
			eecho "I AM NO LONGER ASKING."
			kill -9 "$qemu_pid"
		done
	fi
	eecho "KILLING VFSD"
	kill "$vfsd_pid"
	eecho "BYE!"
}

qgasend() {
	test -n "$1" || die "empty QGA command"
	eecho "sending qga command: $1"
	response="$( (printf '%s\n' "$1"; sleep 2) | socat - UNIX-CONNECT:"$qga_sock")"
	eecho "received qga response: $response"
	printf '%s' "$response" 
}

descramble_qga_status() {
	test -n "$1" || die "empty QGA status"
	err_data="$(
		printf '%s' "$1" |
			jq -r '.return."err-data"' |
			base64 -d |
			$iconv -f IBM866 -t UTF-8
		)" || eecho "Error descrambling QGA status!"
	eecho
	eecho "QGA error data: $err_data"
	eecho
}

trap cleanup EXIT
trap cleanup HUP
trap cleanup INT

if [ "$(realpath "$0" 2>/dev/null)" != "$(realpath "./win/scripts/build-in-vm.sh" 2>/dev/null)" ]
then
	eecho "Must run from root of repo!!"
	exit 9
fi

# Sometimes virtiofsd is in /usr/libexec which is usually not in PATH
export PATH="$PATH:/usr/libexec"

eecho "CHECKING DEPS"

for dep in qemu-system-x86_64 virtiofsd jq socat
do
	if ! command -v "$dep"
	then
		eecho "'$dep' is missing, go install it!"
		eecho "Hint: some executables may get installed into "
		eecho "/usr/libexec which is usually not in PATH, "
		eecho "but I will check there is as well!"
		exit 2
	fi
done

if command -v "gnu-iconv"
then
	eecho "Found gnu iconv..."
	iconv="gnu-iconv"
else
	iconv="iconv"
fi

if ! ( $iconv -l | grep 'IBM866' > /dev/null )
then
	eecho
	eecho
	eecho
	eecho 'WARNING your iconv does not support IBM866. '
	eecho 'This is not critical, but it means that I will not be able '
	eecho 'to descramble error messages reported by qemu guest agent, '
	eecho 'which might make debugging a pain. '
	eecho 'To fix this, go install gnu iconv!'
	eecho '(the package is called "gnu-libiconv" on Alpine Linux) '
	eecho
	eecho
	eecho
fi

eecho "CHECKING VM IMAGE"
if ! [ -r "$qcow" ]
then
	eecho "qcow vm image at '$qcow' is missing or inaccessible, bye!"
	exit 3
fi

eecho "MAKING TMP DIRS"
mkdir -p "$(dirname "$qmon_sock")" || die "Could not mkdir for '$qmon_sock'"
mkdir -p "$(dirname "$vfsd_sock")" || die "Could not mkdir for '$vfsd_sock'"
mkdir -p "$(dirname "$qga_sock")" || die "Could not mkdir for '$qga_sock'"
mkdir -p "$(dirname "$overlay")" || die "Could not mkdir for '$overlay'"
mkdir -p "$(dirname "$qemu_pid_file")" || die "Could not mkdir for '$qemu_pid_file'"

eecho "MAKING OVERLAY qcow2"

if [ -r "$overlay" ]
then
	eecho "OVERLAY EXISTS, NOT MAKING NEW ONE."
else
	qemu-img create -f qcow2 -F qcow2 -b "$qcow" "$overlay" \
		|| die "Could not make overlay qcow2"

	qemu-img info "$overlay" \
		|| die "Could not get info on overlay qcow2 -- is it borked?"
fi

eecho "STARTING VIRTIOFSD"
virtiofsd \
	--socket-path="$vfsd_sock" \
	--shared-dir="." \
	--sandbox=none \
	&

vfsd_pid=$!

eecho "Waiting for virtiofsd to get settled..."
sleep 1

test -n "$vfsd_pid" || die "Virtiofsd did not launch, bye!"
kill -0 "$vfsd_pid" || die "Virtiofsd ded, bye!"

eecho "STARTING VM"
qemu-system-x86_64 \
	\
	`# Generic` \
	\
	-m 4G `# Be careful setting too low otherwise virtiofs wont work lol` \
	-smp 4 \
	-enable-kvm \
	-cpu host \
	-drive file="$overlay",format=qcow2 \
	`#-nographic` \
	-display sdl \
	\
	`# network` \
	\
	-netdev user,id=net0,restrict=yes \
	-device e1000,netdev=net0 \
	\
	`# virtiofs` \
	\
	-chardev socket,id=char0,path="$vfsd_sock" \
	-device vhost-user-fs-pci,queue-size=1024,chardev=char0,tag=my_virtiofs \
	-object memory-backend-file,id=mem,size=4G,mem-path=/dev/shm,share=on \
	-numa node,memdev=mem \
	\
	`# etc ` \
	\
	-chardev socket,path="$qga_sock",server,nowait,id=qga0 \
	-device virtio-serial \
	-device virtserialport,chardev=qga0,name=org.qemu.guest_agent.0 \
	-monitor unix:"$qmon_sock",server,nowait \
	-serial none \
	-pidfile "$qemu_pid_file" \
	-daemonize

for i in $(seq 5)
do
	eecho "Making sure VM has started... ($i/5)"

	qemu_pid="$(cat "$qemu_pid_file")"
	if [ -z "$qemu_pid" ]
	then
		die "Could not read qemu pidfile, bye!"
	fi

	if ! kill -0 "$qemu_pid"
	then
		die "qemu died, bye!"
	fi

	sleep 1
done

eecho "Looks like VM is running."

eecho "WAITING FOR QEMU GUEST AGENT"
qga_ok=""

for i in $(seq 100)
do
	if [ -n "$(qgasend '{"execute":"guest-ping"}')" ]
	then
		eecho "Guest agent is ready."
		qga_ok="yes"
		break
	fi
	eecho "Waiting for guest-agent, try $i/100"
	sleep 1
done

if [ -z "$qga_ok" ]
then
	die "qemu agent did not start, bye"
fi

eecho "MOUNTING Z AND WAITING 2 seconds..."
qgasend '{"execute": "guest-exec", "arguments":{"path":"cmd.exe","arg":["/c","sc start VirtioFsSvc"], "capture-output": true}}'
eecho "Ideally should poll command and wait for result but whatever..."
sleep 2

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


