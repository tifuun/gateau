#!/bin/sh

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

if [ -z "$CONTAINER_ACTION" ]
then

	if [ -z "$(podman images -q gateau-cuda11)" ]
	then
		outside_build_cuda11 || exit 5
	else
		echo "CUDA11 CONTAINER ALREADY PRESENT"
	fi

	if [ -z "$(podman images -q gateau-cuda12)" ]
	then
		outside_build_cuda12 || exit 5
	else
		echo "CUDA12 CONTAINER ALREADY PRESENT"
	fi

	outside_testall || exit 6

else
	$CONTAINER_ACTION
fi
