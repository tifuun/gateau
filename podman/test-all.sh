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

inside_testall() {
	# This function runs IN THE CONTAINER
	# and tests gateau
	#
	if [ -z "$(podman images -q gateau-cuda11)" ]
	then
		echo "CUDA11 IMAGE NOT PRESENT!!"
		exit 9
	fi

	if [ -z "$(podman images -q gateau-cuda12)" ]
	then
		echo "CUDA12 IMAGE NOT PRESENT!!"
		exit 9
	fi
	
	set -e

	cuda_name=$(
		nvcc --version |
			grep Build |
			cut -d . -f 1 |
			sed 's/Build //' |
			tr -d '_'
		)
	
	for venv in /venv*
	do
		py_name=$(echo "$venv" | tr -cd '0-9.')

		set +e
		"${venv}/bin/pip" install /gateau
		exit_code=$?
		set -e

		echo "$exit_code" > "/output/${cuda_name}_${py_name}.exitcode"
	done
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

outside_testall() {
	# This function runs ON THE HOST
	# and launches containers to test gateau
	
	set -e
	
	mkdir -p podman/output
	rm -rf podman/output/*

	for cuda in cuda11 cuda12
	do
		podman run \
			--rm \
			--init \
			-v ./:/gateau:ro \
			-v ./podman/output:/output:rw \
			-e CONTAINER_ACTION='inside_testall' \
			--workdir /gateau \
			"gateau-${cuda}" \
			/gateau/podman/test-all.sh
	done
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

	set -e

	case "$1" in
		test)
			outside_testall
			;;
		build)
			outside_build_images
			;;
		pull)
			outside_pull_images
			;;
		push)
			outside_push_image gateau-cuda11
			outside_push_image gateau-cuda12
			;;
		*)
			echo "Usage: $0 <test|build|pull|push>"
			exit 9
			;;
	esac

else
	$CONTAINER_ACTION
fi
