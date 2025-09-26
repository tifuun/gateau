#!/bin/sh

# Subcommands
#
# `test`
# 	Run the full test suite
#
# `test11`
# 	Run full test suite only on cuda11 container
#
# `ruff`
# 	Run ruff checker in `cicd` container
#
# `build`
# 	Build all missing container images
#
# `pull`
# 	Pull all container images from reg.stratal.systems
# 	using `public:public` credentials,
# 	overriding existing ones
#
# `push`
# 	Push all container images to reg.stratal.systems,
# 	overriding existing ones. Must `podman login` first!
#
# `wheel`
# 	Build wheel using `cicd` container.
# 	This will build libgateau with DYNAMICALLY linked
# 	libgsl!
#
# `staticwheel`
# 	Build wheel using `cicd` container.
# 	This will build libgateau with STATICALLY linked
# 	libgsl!
#
# `docs`
# 	Builds html docs. Output is under `./podman/output/docs`.
#
# `ocs_with_coverage`
# 	Builds html docs. Output is under `./podman/output/docs`.
# 	Also includes htmlvoc report under `./podman/output/docs/htmlcov`.
#
# `teststatic`
# 	Verify that the `libgateau.so` inside
# 	the wheel file in the `dist` directory
# 	has libgsl linked STATICALLY
#
# `test-wheel`
# 	Install gateau from wheel in `cuda11bare` container
# 	and run tests
#
# `tpypi`
# 	Upload built wheel to testpypi.
# 	Must set `TWINE_USERNAME` and `TWINE_PASSWORD`
# 	env variables for this to work!
# 	If using token,
# 	set `TWINE_USERNAME` to `__token__`
# 	and `TWINE_PASSWORD` to the token.
# 	Must run `wheel` subcomand first to build the wheel!
#
# `test-tpypi`
# 	Download wheel from testpypi, install, and test
#
# `pypi`
# 	Upload built wheel to the real pypi
# 	Must run `wheel` subcomand first to build the wheel!
#
# `test-pypi`
# 	Download wheel from the real pypi, install, and test.
# 	Must set `TWINE_USERNAME` and `TWINE_PASSWORD`
# 	env variables for this to work!
# 	If using token,
# 	set `TWINE_USERNAME` to `__token__`
# 	and `TWINE_PASSWORD` to the token.

# Environment variables
#
# GATEAU_TEST_VENVS
# 	Which of the available venvs (i.e. python versions)
# 	to use to test gateau.
# 	Defaults to all venvs available in container.
# 	Has effect for `test` and `test11` subcommands.
# 	The value is whitespace-separated list of paths.
# 	Example:
# 	GATEAU_TEST_VENVS="/venv3.9 /venv3.13"
#
# GATEAU_EDITABLE_INSTALL
# 	Whether or not to use editable install for testing.
# 	Set to emptystring or unset for non-editable install
# 	(Default), set to non-empty string for
# 	editable install.
# 	Has effect for `test` and `test11` subcommands.
#
# GATEAU_DISABLE_GPU
# 	Set this to a non-empty string in order
# 	to run podman without the `--gpus=all` flag.
# 	Default is to run with gpu.



# Internal environment variables
#
# These are used by the script to communicate with itself
# do not try to change these
#
# CONTAINER_ACTION
# 	Used internally by this script to tell itself what to do once
# 	it runs inside the container. This is simply a function name
# 	(though it could also be any command) 
# 	that is executed directly.
#
# TWINE_REPOSITORY
# TWINE_USERNAME
# TWINE_PASSWORD
# 	These are standard variables interpreted by twine
# 	directly. The script changes them depending on
# 	whether you're using the `pypi` or `tpypi` subcommand
#
# GATEAU_PIP_FLAGS
#	Used to change index url for `test_tpypi` subcommand,
#	no effect on other subcommands.
#



# Images
#
# gateau-cuda11
# 	Image based on nvidia's cuda11 with some pythons
# 	and gateau's build deps.
# 	For testing.
#
# gateau-cuda12
# 	Image based on nvidia's cuda12 with some pythons
# 	and gateau's build deps
# 	For testing.
#
# gateau-cicd
# 	Like cuda11, but with only one python, ruff, build, and twine.
# 	Used for CI and building wheel.
# 	Cuda11 because mitakihara does not support cuda12
#

if [ "$(realpath "$0" 2>/dev/null)" != "$(realpath "./podman/test-all.sh" 2>/dev/null)" ]
then
	echo "Must run from root of repo!!"
	exit 9
fi

if [ -n "$GATEAU_DISABLE_GPU" ]
then
	podman_gpu=''
else
	podman_gpu='--gpus=all'
fi

# echo to stderr
eecho() {
	echo "$@" >&2
}

# echo to stderr and exit with error
die() {
	eecho "$@"
	exit 1
}

outside_check_podman() {
	podman --version >/dev/null || die "Podman not installed, bye!"
}

# Yell at user if the podman storage driver is not overlayfs
outside_check_podman_storage_driver() {
	driver="$(podman info --format '{{.Store.GraphDriverName}}')" ||
		die 'Could not check podman storage driver'
	
	if [ "$driver" != "overlay" ]
	then
		echo 'Podman storage driver is not overlay!!'
		echo 'Fix it by changing the podman storage driver to `overlay`'
		echo '(see https://github.com/stratal-systems/wiki/blob/main/wiki/podman-storage-driver.md)'
		echo 'or set GATEAU_IGNORE_STORAGE_DRIVER_YES_I_KNOW_WHAT_I_AM_DOING=1 to skip this check.'

		if [ -z "$GATEAU_IGNORE_STORAGE_DRIVER_YES_I_KNOW_WHAT_I_AM_DOING" ]
		then
			die "Aborting due to weird storage driver."
		fi
	fi
}

outside_build_cuda11() {
	set -e

	echo BUILDING CONTAINER FOR CUDA11
	podman build \
		--file ./podman/cuda11.Dockerfile -t gateau-cuda11 ./podman
	sync
}

outside_build_cuda12() {
	set -e

	echo BUILDING CONTAINER FOR CUDA12
	podman build \
		--file ./podman/cuda12.Dockerfile -t gateau-cuda12 ./podman
	sync
}

outside_build_cicd() {
	set -e

	echo BUILDING CICD CONTAINER
	podman build \
		--file ./podman/cicd.Dockerfile -t gateau-cicd ./podman
	sync
}

outside_build_cuda12bare() {
	set -e

	echo BUILDING CONTAINER FOR CUDA12bare
	podman build \
		--file ./podman/cuda12bare.Dockerfile -t gateau-cuda12bare ./podman
	sync
}

outside_build_cuda11bare() {
	set -e

	echo BUILDING CONTAINER FOR CUDA11bare
	podman build \
		--file ./podman/cuda11bare.Dockerfile -t gateau-cuda11bare ./podman
	sync
}

inside_testall() {
	# This function runs IN THE CONTAINER
	# and tests gateau
	#
	
	set -e

	cuda_name=$(
		nvcc --version |
			grep Build |
			cut -d . -f 1 |
			sed 's/Build //' |
			tr -d '_'
		)
	
	if [ -n "$GATEAU_TEST_VENVS" ]
	then
		venvs="$GATEAU_TEST_VENVS"
	else
		venvs=/venv*
	fi

	if [ -n "$GATEAU_TEST_EDITABLE_INSTALL" ]
	then
		pipinstallflags='-e'
		echo 'USING EDITABLE INSTALL!!'
	else
		pipinstallflags=''
	fi
	
	for venv in $venvs
	do
		py_name=$(echo "$venv" | tr -cd '0-9.')

		echo "INSTALLING..."
		set +e
		"${venv}/bin/pip" install $pipinstallflags  /gateau
		exit_code=$?
		set -e

		echo "$exit_code" > \
			"/output/${cuda_name}_${py_name}.install.exitcode"

		echo "TESTING..."
		set +e
		"${venv}/bin/python" -m unittest
		exit_code=$?
		set -e

		echo "$exit_code" > \
			"/output/${cuda_name}_${py_name}.test.exitcode"
	done
}

inside_test_wheel() {

	wheel=$(ls dist/*.whl | head -n 1)

	set -e

	if [ -z "$wheel" ]
	then
		echo "No wheels found in dist folder, aborting"
		exit 2
	fi

	for venv in /venv3.13
	do
		py_name=$(echo "$venv" | tr -cd '0-9.')

		echo "Installing wheel $wheel for python $py_name"
		set +e
		"${venv}/bin/pip" install "$wheel"
		exit_code=$?
		set -e

		echo "$exit_code" > \
			"/output/${cuda_name}_${py_name}.install.exitcode"

		echo "TESTING..."
		set +e
		"${venv}/bin/python" -m unittest
		exit_code=$?
		set -e

		echo "$exit_code" > \
			"/output/${cuda_name}_${py_name}.test.exitcode"
	done

	
	#set -e
	#"/venv3.12/bin/pip" -v -v -v -v -v install scikit-build-core setuptools

	#set +e
	#"/venv3.12/bin/pip" -v -v -v -v -v install -i https://test.pypi.org/simple/ gateau
	#exit_code=$?
	#set -e

	#echo "$exit_code" > \
	#	"/output/testpypi.install.exitcode"
	true
}

outside_build_images() {
	if [ -n "$(podman images -q gateau-cuda11)" ]
	then
		echo "CUDA11 CONTAINER ALREADY PRESENT"
	else
		outside_build_cuda11
	fi

	if [ -n "$(podman images -q gateau-cuda12)" ]
	then
		echo "CUDA12 CONTAINER ALREADY PRESENT"
	else
		outside_build_cuda12
	fi

	if [ -n "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD CONTAINER ALREADY PRESENT"
	else
		outside_build_cicd
	fi

	if [ -n "$(podman images -q gateau-cuda11bare)" ]
	then
		echo "CUDA11BARE CONTAINER ALREADY PRESENT"
	else
		outside_build_cuda11bare
	fi

}

outside_pull_images() {
	outside_pull_image gateau-cuda11
	outside_pull_image gateau-cuda12
	outside_pull_image gateau-cuda11bare
	outside_pull_image gateau-cicd
}

inside_ruff() {

	ruff check . \
		--output-file /output/ruff.txt

	echo "Ran ruff, output is int podman/output/ruff.txt"
}

inside_docs() {
	set -e

	echo ACTIVATING VENV
	. /venv3.13/bin/activate

	echo BUILDING DOCS
	./scripts/GenerateDocs.py 

	echo COPYING ARTIFACT
	cp -r --reflink=auto ./docs -t "/output/"

	echo DONE
}

inside_docs_with_coverage() {
	set -e

	echo ACTIVATING VENV
	. /venv3.13/bin/activate

	echo BUILDING DOCS
	./scripts/GenerateDocs.py 

	echo ALSO BUILDING COVERAGE REPORT
	pip install -e .
	coverage run -m unittest ||
		echo 'looks like tests failed, publishing coverage anyway.'
	coverage html

	echo COPYING ARTIFACT
	cp -r --reflink=auto ./docs -t "/output"
	cp -r --reflink=auto ./htmlcov -t "/output/docs"

	echo DONE
}

inside_staticwheel() {
	export DEBIAN_FRONTEND=noninteractive
	set -e
	apt purge -y libgsl-dev

	# Go read `MAINTAINERS.md` on why this line is needed
	rm -rf /usr/local/lib/libgsl.so*

	apt install -y wget autotools-dev autoconf libtool gpg
	./scripts/build-gsl.sh
	inside_wheel
	# TODO test with LDD that its actually static
}

inside_wheel() {
	set -e

	echo "Building wheel..."

	rm -rf dist/*

	# Use this for standard glibc (in cuda12 container)
	#glibc_ver=$(ldd --version | grep 'GNU libc' | head -n 1 | rev | cut -d ' ' -f 1 | rev | tr '.' '_')

	glibc_ver=$(ldd --version | grep 'Ubuntu GLIBC' | head -n 1 | rev | cut -d ' ' -f 1 | rev | tr '.' '_')

	if [ -z "$glibc_ver" ]
	then
		echo "Could not determine glibc ver, aborting."
		exit 2
	fi

	echo "found glibc $glib_ver"

	/venv3.13/bin/python -m build
	
	for f in dist/*
	do
		renamed="$(echo "$f" | sed -e "s|linux_x86_64|manylinux_${glibc_ver}_x86_64|" -e "s|cp313-cp313|cp39.cp310.cp311.cp312.cp313-none|")"
		if [ "$renamed" != "$f" ]
		then
			mv "$f" "$renamed"
		fi
	done
}

inside_checkstatic() {
	set -e

	echo "Checking whether wheel is static..."

	echo "Looking for wheel..."

	wfile=$(find ./dist -name '*.whl' | head -n 1)
	if [ -z "$wfile" ]
	then
		echo "Could not find wheel."
		exit 2
	fi

	mkdir -p /tmp/unpack
	unzip "$wfile" -d /tmp/unpack

	ldd_output=$(ldd /tmp/unpack/gateau/libgateau.so)

	echo "LDD output:"
	echo "$ldd_output"

	if ! ( echo "$ldd_output" | grep libgsl )
	then
		echo "Looks like it's static!!"
	else
		echo "Looks like it's not static!!"
		exit 3
	fi
}

outside_wheel() {

	if [ -z "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e

	mkdir -p ./dist
	
	podman run \
		--rm \
		--init \
		`# $podman_gpu ` \
		-v ./:/gateau:O \
		-v ./dist:/gateau/dist:rw \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_wheel' \
		--workdir /gateau \
		"gateau-cicd" \
		/gateau/podman/test-all.sh
}

outside_staticwheel() {

	if [ -z "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e

	mkdir -p ./dist
	
	podman run \
		--rm \
		--init \
		`# $podman_gpu ` \
		-v ./:/gateau:O \
		-v ./dist:/gateau/dist:rw \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_staticwheel' \
		--workdir /gateau \
		"gateau-cicd" \
		/gateau/podman/test-all.sh
}

outside_docs() {

	if [ -z "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e

	podman run \
		--rm \
		--init \
		`# $podman_gpu ` \
		-v ./:/gateau:O \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_docs' \
		--workdir /gateau \
		"gateau-cicd" \
		/gateau/podman/test-all.sh
}

outside_docs_with_coverage() {

	if [ -z "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e

	podman run \
		--rm \
		--init \
		`# $podman_gpu ` \
		-v ./:/gateau:O \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_docs_with_coverage' \
		--workdir /gateau \
		"gateau-cicd" \
		/gateau/podman/test-all.sh
}

outside_checkstatic() {

	if [ -z "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e

	mkdir -p ./dist
	
	podman run \
		--rm \
		--init \
		`# $podman_gpu ` \
		-v ./:/gateau:O \
		-v ./dist:/gateau/dist:rw \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_checkstatic' \
		--workdir /gateau \
		"gateau-cicd" \
		/gateau/podman/test-all.sh
}


outside_test_wheel() {

	if [ -z "$(podman images -q gateau-cuda11bare)" ]
	then
		outside_build_cuda11bare
	fi
	
	set -e
	
	podman run \
		--rm \
		--init \
		$podman_gpu \
		-v ./:/gateau:O \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_test_wheel' \
		`#-e GATEAU_TEST_VENVS` \
		`#-e GATEAU_TEST_EDITABLE_INSTALL` \
		--workdir /gateau \
		"gateau-cuda11bare" \
		/gateau/podman/test-all.sh
}

outside_testall() {
	# This function runs ON THE HOST
	# and launches containers to test gateau
	#
	if [ -z "$(podman images -q gateau-cuda11)" ]
	then
		echo "CUDA11 IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi

	if [ -z "$(podman images -q gateau-cuda12)" ]
	then
		echo "CUDA12 IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e
	
	rm -rf podman/output/*.exitcode

	for cuda in cuda11 cuda12
	do
		podman run \
			--rm \
			--init \
			$podman_gpu \
			-v ./:/gateau:O \
			-v ./podman/output:/output:rw \
			-e CONTAINER_ACTION='inside_testall' \
			-e GATEAU_TEST_VENVS \
			-e GATEAU_TEST_EDITABLE_INSTALL \
			--workdir /gateau \
			"gateau-${cuda}" \
			/gateau/podman/test-all.sh
	done
}

outside_test11() {
	if [ -z "$(podman images -q gateau-cuda11)" ]
	then
		echo "CUDA11 IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi

	set -e
	
	rm -rf podman/output/*.exitcode

	podman run \
		--rm \
		--init \
		$podman_gpu \
		-v ./:/gateau:O \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_testall' \
		-e GATEAU_TEST_VENVS \
		-e GATEAU_TEST_EDITABLE_INSTALL \
		--workdir /gateau \
		"gateau-cuda11" \
		/gateau/podman/test-all.sh
}

outside_ruff() {
	if [ -z "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi

	set -e
	
	rm -rf podman/output/*.exitcode

	podman run \
		--rm \
		--init \
		-v ./:/gateau:O \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_ruff' \
		--workdir /gateau \
		"gateau-cicd" \
		/gateau/podman/test-all.sh
}

outside_pull_image() {
	ref="$1"

	set -e
	podman pull --creds 'public:public' reg.stratal.systems/"$ref"
	podman tag reg.stratal.systems/"$ref" "$ref"
}

outside_push_image() {
	ref="$1"

	set -e
	podman push "$ref" reg.stratal.systems/"$ref"
}

outside_pypi() {

	if [ -z "$(podman images -q gateau-cicd)" ]
	then
		echo "CICD IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e

	if [ "$(ls dist | wc -l)" -le 0 ]
	then
		echo "Nothing in dist directory! Did you build wheel?"
		exit 10
	fi

	if [ -z "$TWINE_USERNAME" ]
	then
		echo "TWINE_USERNAME not set, aborting!"
		exit 11
	fi

	if [ -z "$TWINE_PASSWORD" ]
	then
		echo "TWINE_PASSWORD not set, aborting!"
		exit 12
	fi
	
	podman run \
		--rm \
		--init \
		`# $podman_gpu ` \
		-v ./:/gateau:O \
		-v ./dist:/gateau/dist:rw \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_pypi' \
		-e TWINE_REPOSITORY \
		-e TWINE_USERNAME \
		-e TWINE_PASSWORD \
		--workdir /gateau \
		"gateau-cicd" \
		/gateau/podman/test-all.sh
}

inside_pypi() {
	/venv3.13/bin/twine upload \
		--verbose \
		--skip-existing \
		--non-interactive \
		dist/*
}

inside_test_pypi() {
	
	set -e

	cuda_name=$(
		nvcc --version |
			grep Build |
			cut -d . -f 1 |
			sed 's/Build //' |
			tr -d '_'
		)
	
	if [ -n "$GATEAU_TEST_VENVS" ]
	then
		venvs="$GATEAU_TEST_VENVS"
	else
		venvs=/venv*
	fi

	for venv in $venvs
	do
		py_name=$(echo "$venv" | tr -cd '0-9.')

		echo "INSTALLING..."
		set +e
		"${venv}/bin/pip" install gateau \
			$GATEAU_PIP_FLAGS \
			--only-binary=:all:
		exit_code=$?
		set -e

		echo "$exit_code" > \
			"/output/${cuda_name}_${py_name}.install.exitcode"

		echo "TESTING..."
		set +e
		"${venv}/bin/python" -m unittest
		exit_code=$?
		set -e

		echo "$exit_code" > \
			"/output/${cuda_name}_${py_name}.test.exitcode"
	done
}

outside_test_pypi() {

	if [ -z "$(podman images -q gateau-cuda11bare)" ]
	then
		outside_build_cuda11bare
	fi
	
	set -e
	
	podman run \
		--rm \
		--init \
		$podman_gpu \
		-v ./:/gateau:O \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_test_pypi' \
		-e GATEAU_TEST_VENVS \
		-e GATEAU_PIP_FLAGS \
		`#-e GATEAU_TEST_EDITABLE_INSTALL` \
		--workdir /gateau \
		"gateau-cuda11bare" \
		/gateau/podman/test-all.sh
}


if [ -z "$CONTAINER_ACTION" ]
then

	mkdir -p podman/output

	set -e

	case "$1" in
		test)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_testall
			} 2>&1 | tee podman/output/log.txt
			;;
		test11)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_test11
			} 2>&1 | tee podman/output/log.txt
			;;
		ruff)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_ruff
			} 2>&1 | tee podman/output/log.txt
			;;
		build)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_build_images
			} 2>&1 | tee podman/output/log.txt
			;;
		pull)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_pull_images
			} 2>&1 | tee podman/output/log.txt
			;;
		push)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_push_image gateau-cuda11
				outside_push_image gateau-cuda11bare
				outside_push_image gateau-cuda12
				outside_push_image gateau-cicd
			} 2>&1 | tee podman/output/log.txt
			;;
		wheel)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_wheel
			} 2>&1 | tee podman/output/log.txt
			;;
		staticwheel)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_staticwheel
			} 2>&1 | tee podman/output/log.txt
			;;
		docs)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_docs
			} 2>&1 | tee podman/output/log.txt
			;;
		docs_with_coverage)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_docs_with_coverage
			} 2>&1 | tee podman/output/log.txt
			;;
		checkstatic)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_checkstatic
			} 2>&1 | tee podman/output/log.txt
			;;
		test-wheel)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				outside_test_wheel
			} 2>&1 | tee podman/output/log.txt
			;;
		tpypi)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				export TWINE_REPOSITORY=testpypi
				outside_pypi
			} 2>&1 | tee podman/output/log.txt
			;;
		test-tpypi)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				# extra-index-url is needed because
				# dependencies (numpy, etc)
				# are not on testpypi
				export GATEAU_PIP_FLAGS="
				--index-url https://test.pypi.org/simple/
				--extra-index-url https://pypi.org/simple/
				"
				outside_test_pypi
			} 2>&1 | tee podman/output/log.txt
			;;
		pypi)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				export TWINE_REPOSITORY=pypi
				outside_pypi
			} 2>&1 | tee podman/output/log.txt
			;;
		test-pypi)
			{
				outside_check_podman
				outside_check_podman_storage_driver
				export GATEAU_PIP_FLAGS=""
				outside_test_pypi
			} 2>&1 | tee podman/output/log.txt
			;;
		*)
			echo "Usage: $0 <test|test11|ruff|build|pull|push|wheel|staticwheel|docs|docs_with_coverage|checkstatic|test-wheel|tpypi|test-tpypi|pypi|test-pypi>"
			exit 9
			;;
	esac

else
	$CONTAINER_ACTION
fi

