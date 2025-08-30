#!/bin/sh

if [ "$(realpath "$0" 2>/dev/null)" != "$(realpath "./podman/test-all.sh" 2>/dev/null)" ]
then
	echo "Must run from root of repo!!"
	exit 9
fi

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

outside_build_cuda12bare() {
	set -e

	echo BUILDING CONTAINER FOR CUDA12bare
	podman build \
		--file ./podman/cuda12bare.Dockerfile -t gateau-cuda12bare ./podman
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
	
	for venv in $venvs
	do
		py_name=$(echo "$venv" | tr -cd '0-9.')

		echo "INSTALLING..."
		set +e
		"${venv}/bin/pip" install /gateau
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

inside_test_test_pypi() {
	
	set -e
	"/venv3.12/bin/pip" -v -v -v -v -v install scikit-build-core setuptools

	set +e
	"/venv3.12/bin/pip" -v -v -v -v -v install -i https://test.pypi.org/simple/ gateau
	exit_code=$?
	set -e

	echo "$exit_code" > \
		"/output/testpypi.install.exitcode"
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

}

outside_pull_images() {
	if [ -n "$(podman images -q gateau-cuda11)" ]
	then
		echo "CUDA11 CONTAINER ALREADY PRESENT"
	else
		outside_pull_image gateau-cuda11
	fi

	if [ -n "$(podman images -q gateau-cuda12)" ]
	then
		echo "CUDA12 CONTAINER ALREADY PRESENT"
	else
		outside_pull_image gateau-cuda12
	fi

}

inside_wheel_cuda12() {
	set -e

	rm -rf dist/*
	glibc_ver=$(ldd --version | grep 'GNU libc' | head -n 1 | rev | cut -d ' ' -f 1 | rev | tr '.' '_')
	test "$glibc_ver"
	/venv3.12/bin/python -m build
	rename --verbose linux_x86_64 "manylinux_${glibc_ver}_x86_64" dist/*
}

outside_wheel_cuda12() {

	if [ -z "$(podman images -q gateau-cuda12)" ]
	then
		echo "CUDA12 IMAGE NOT PRESENT!!"
		echo "Please run $0 build or $0 pull"
		exit 9
	fi
	
	set -e
	
	podman run \
		--rm \
		--init \
		--gpus=all \
		-v ./:/gateau:ro \
		-v ./dist:/gateau/dist:rw \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_wheel_cuda12' \
		-e GATEAU_TEST_VENVS \
		--workdir /gateau \
		"gateau-cuda12" \
		/gateau/podman/test-all.sh
}

outside_test_test_pypi() {

	if [ -z "$(podman images -q gateau-cuda12bare)" ]
	then
		outside_build_cuda12bare
	fi
	
	set -e
	
	podman run \
		--rm \
		--init \
		--gpus=all \
		-v ./:/gateau:ro \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_test_test_pypi' \
		-e GATEAU_TEST_VENVS \
		--workdir /gateau \
		"gateau-cuda12bare" \
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
			--gpus=all \
			-v ./:/gateau:ro \
			-v ./podman/output:/output:rw \
			-e CONTAINER_ACTION='inside_testall' \
			-e GATEAU_TEST_VENVS \
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
		--gpus=all \
		-v ./:/gateau:ro \
		-v ./podman/output:/output:rw \
		-e CONTAINER_ACTION='inside_testall' \
		--workdir /gateau \
		"gateau-cuda11" \
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

if [ -z "$CONTAINER_ACTION" ]
then

	mkdir -p podman/output

	set -e

	case "$1" in
		wheel-cuda12)
			{
				outside_wheel_cuda12
			} 2>&1 | tee podman/output/log.txt
			;;
		test-test-pypi)
			{
				outside_test_test_pypi
			} 2>&1 | tee podman/output/log.txt
			;;
		test)
			{
				outside_testall
			} 2>&1 | tee podman/output/log.txt
			;;
		test11)
			{
				outside_test11
			} 2>&1 | tee podman/output/log.txt
			;;
		build)
			{
				outside_build_images
			} 2>&1 | tee podman/output/log.txt
			;;
		pull)
			{
				outside_pull_images
			} 2>&1 | tee podman/output/log.txt
			;;
		push)
			{
				outside_push_image gateau-cuda11
				outside_push_image gateau-cuda12
			} 2>&1 | tee podman/output/log.txt
			;;
		*)
			echo "Usage: $0 <test|build|pull|push|wheel-cuda12|test-test-pypi>"
			exit 9
			;;
	esac

else
	$CONTAINER_ACTION
fi
