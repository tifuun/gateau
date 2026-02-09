#!/bin/sh

# echo to stderr
eecho() {
	echo "$@" >&2
}

# echo to stderr and exit with error
die() {
	eecho "$@"
	exit 1
}

load_paths() {
	. ./win/scripts/paths.sh \
		|| die 'could not source paths.sh, correct PWD?'
}


enforce_pwd() {
	if [ -z "$1" ]
	then
		die "enforce_pwd called with empty arg"
	fi

	if [ "$(realpath "$0" 2>/dev/null)" != "$(realpath "$1" 2>/dev/null)" ]
	then
		die "Must run from root of repo!!"
	fi
}

path_libexec() {
	# Sometimes virtiofsd is in
	# /usr/libexec which is usually not in PATH
	export PATH="$PATH:/usr/libexec"
}

check_deps() {
	eecho "CHECKING DEPS"
	for dep in "$@"
	do
		if ! command -v "$dep"
		then
			eecho "command '$dep' is missing, go install it!"
			eecho
			eecho "HINT: some executables may get installed into "
			eecho "/usr/libexec which is usually not in PATH, "
			eecho "but I will check there as well!"
			eecho
			eecho "HINT: 'win/packages' contains scripts "
			eecho "for various distros that install all "
			eecho "needed packages automatically. "
			exit 2
		fi
	done
}

check_sdl_ui() {
	has_sdl="$(qemu-system-x86_64 -display help | grep '^sdl')" \
		|| die "qemu SDL ui is missing, go install it!"
	if [ -z "has_sdl" ]
	then
		die "qemu SDL ui is missing, go install it!"
	fi
}

check_deps_qemu() {
	check_deps qemu-system-x86_64 virtiofsd jq socat qemu-img unzip
	check_sdl_ui
}


find_iconv() {
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
}

start_virtiofsd() {
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

wait_vm_start() {
	eecho "WAITING FOR VM TO START"
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
}

wait_vm_finish_infinite() {
	while true
	do
		qemu_pid="$(cat "$qemu_pid_file")"
		if [ -z "$qemu_pid" ]
		then
			die "Could not read qemu pidfile, bye!"
		fi

		if ! kill -0 "$qemu_pid"
		then
			die "qemu died, bye!"
		fi

		sleep 5
		eecho "vm is still running..."
	done
}

wait_qga() {
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
}

mount_virtiofs() {
	eecho "MOUNTING Z AND WAITING 2 seconds..."
	qgasend '{"execute": "guest-exec", "arguments":{"path":"cmd.exe","arg":["/c","sc start VirtioFsSvc"], "capture-output": true}}'
	eecho "Ideally should poll command and wait for result but whatever..."
	sleep 2
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

start_qemu() {
	eecho "STARTING VM"

	if [ -z "$1" ]
	then
		eecho "start_qemu ran without arg!"
		eecho "I don't what what image to use!"
		exit 1
	fi
	qcow_run_image="$1"

	qemu-system-x86_64 \
		\
		`# Generic` \
		\
		-m 4G `# Be careful setting too low otherwise virtiofs wont work lol` \
		-smp 4 \
		-enable-kvm \
		-cpu host \
		-drive file="$qcow_run_image",format=qcow2 \
		`#-nographic` \
		-display sdl \
		\
		`# network` \
		\
		`#-netdev user,id=net0,restrict=yes` \
		-netdev user,id=net0,restrict=no \
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
		-rtc base=2077-01-01 \
		`# even when setting rtc to localtime, ` \
		`# meson complains about clock skew. ` \
		`# so brute force it by setting it WAY in the future lol.` \
		-daemonize
}

start_qemu_with_overlay() {
	start_qemu "$overlay"
}

start_qemu_with_no_overlay() {
	eecho
	eecho
	eecho
	eecho "Starting qemu with NO OVERLAY!"
	eecho "Changes made to vm image are PERMANENT."
	eecho "vm image is: $qcow"
	eecho
	eecho

	start_qemu "$qcow"
}

check_empty_dist() {
	if [ -n "$(find dist -name '*.whl')" ]
	then
		eecho "'dist' folder is not empty! Please clear out anything in "
		eecho "'dist' to avoid conflicts. "
		exit 5
	fi
}


check_vm_image() {
	eecho "CHECKING VM IMAGE"
	if ! [ -r "$qcow" ]
	then
		eecho "qcow vm image at '$qcow' is missing or inaccessible!"
		eecho "Install and configure git-annex, then run "
		eecho "'git annex get win/vm/5.final.qcow2' and wait for the download "
		eecho "(40GB + hosted on a slow server, so it'll take a while.)"
		exit 3
	fi
}

check_pipcache() {
	eecho "CHECKING PIPCACHE"
	if ! [ -r "$( find win/pipcache/ -name 'meson_python*.whl' | head -n 1 )" ]
	then
		eecho "Looks like python dependencies in 'win/pipcache' are "
		eecho "missing or inaccessible. "
		eecho "Set up and configure git-annex, then run "
		eecho "git annex get win/pipcache to download them, "
		eecho "or see win/scripts/compile-requirements.sh "
		eecho "and win/scripts/download-wheels.sh "
		eecho "to fetch newer versions from pypi. "
		exit 2
	fi
}

make_tmpdirs() {
	eecho "MAKING TMP DIRS"
	mkdir -p "$(dirname "$qmon_sock")" || die "Could not mkdir for '$qmon_sock'"
	mkdir -p "$(dirname "$vfsd_sock")" || die "Could not mkdir for '$vfsd_sock'"
	mkdir -p "$(dirname "$qga_sock")" || die "Could not mkdir for '$qga_sock'"
	mkdir -p "$(dirname "$overlay")" || die "Could not mkdir for '$overlay'"
	mkdir -p "$(dirname "$qemu_pid_file")" || die "Could not mkdir for '$qemu_pid_file'"
	mkdir -p "$unzip_dir" || die "Could not mkdir for '$unzip_dir'"
	mkdir -p "$pipcache_link_dir" || die "Could not mkdir for '$pipcache_link_dir'"
}

make_overlay() {
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
}

check_kvm_group() {
	if [ -n "$GATEAU_SKIP_KVM_CHECK" ]
	then
		eecho "Skipping kvm permission check!"
		return
	fi

	if [ "$(id -u)" -eq 0 ] || id -nG | grep -qw kvm
	then
		echo "ok: user is root or is in kvm group"
	else
		eecho "User is not root and not in 'kvm' group."
		eecho "qemu will fail."
		eecho "Add yourself to 'kvm' group."
		eecho
		eecho 'HINT: sudo usermod -aG kvm "$(whoami)"'
		eecho 'HINT: su "$(whoami)"'
		eecho
		eecho "Run with GATEAU_SKIP_KVM_CHECK=1 to skip this check."
		exit 1
	fi
}

extract_zip() {
	if [ -z "$1" ]
	then
		die "extract_zip ran without source!"
	fi

	if [ -z "$2" ]
	then
		die "extract_zip ran without dest!"
	fi

	if [ ! -r "$1" ]
	then
		die "File '$1' is missing or not readable!"
	fi

	if [ -d "$2" ]
	then
		eecho "$2 already exists,"
		eecho "skipping extraction!"
		return
	fi

	mkdir -p "$2" \
		|| die "Failed to mkdir unzip dest."

	unzip -n "$1" -d "$2" \
		|| die "Failed to unzip"

}

extract_zip_all() {
	extract_zip \
		"win/dep/hdf5-2.0.0-win-vs2022_cl.zip" \
		"$unzip_dir/hdf5"

	extract_zip \
		"$unzip_dir/hdf5/hdf5/HDF5-2.0.0-win64.zip" \
		"$unzip_dir/hdf5/hdf5/HDF5-2.0.0-win64"

	extract_zip \
		"win/dep/gsl-msvc14-x64.2.3.0.2779.zip" \
		"$unzip_dir/gsl"

	extract_zip \
		"win/dep/libaec-1.0.4-h33f27b4_1.zip" \
		"$unzip_dir/aec"

	extract_zip \
		"win/dep/zlib-win-x64.zip" \
		"$unzip_dir/zlib"

}

copy_pipcache() {
	# Hardlink pipcache into separate directory
	# because windows does not understand symlinks
	cp -l win/pipcache/*.whl "$pipcache_link_dir" \
		|| die "Failed to hardlink pipcache!"
}

