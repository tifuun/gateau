FROM debian:bookworm

ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
ARG GSL_SIG_URL=https://mirror.clientvps.com/gnu/gsl/gsl-2.5.tar.gz.sig
ARG GSL_URL=https://mirror.clientvps.com/gnu/gsl/gsl-2.5.tar.gz
ARG LIBGSL_DEB_URL=https://github.com/stratal-systems/debian-packages/releases/download/v2-2/libgsl-static-pic_2.5.0_amd64.deb
ARG LIBHDF5_DEB_URL=https://github.com/stratal-systems/debian-packages/releases/download/v2-2/libhdf5-static-pic_2.0.0_amd64.deb
WORKDIR /setup

RUN \
	apt update && \
	apt install -y \
		autoconf \
		cmake \
		doxygen  `# For doxy` \
		gcc \
		git \
		git-annex \
		gpg \
		libtool \
		lintian \
		make \
		make \
		meson \
		pkg-config \
		python3.11 \
		python3.11-dev \
		python3.11-venv \
		wget \
		zlib1g-dev \
		&& \
		:

RUN \
	wget "${CUDA_KEYRING_URL}" && \
	dpkg -i cuda-keyring_1.1-1_all.deb && \
	apt-get update && \
	apt-get -y install cuda-toolkit-12-3 && \
	:

ENV PATH=/usr/local/cuda/bin:${PATH}

RUN \
	wget "${LIBGSL_DEB_URL}" && \
	wget "${LIBHDF5_DEB_URL}" && \
	dpkg -i *.deb && \
	:

RUN \
	python3.11 -m venv /venv && \
	. /venv/bin/activate && \
	pip install pip-tools && \
	:

COPY ./pyproject.toml /setup/pyproject.toml
COPY ./meson.build /setup/meson.build
COPY ./meson.options /setup/meson.options
COPY ./README.md /setup/README.md
COPY ./src /setup/src

RUN \
	. /venv/bin/activate && \
	pip-compile --verbose --all-build-deps --all-extras pyproject.toml && \
	pip download -r requirements.txt -d pipcache --only-binary=:all: && \
	pip install -r requirements.txt && \
	pip install build twine && \
	:

WORKDIR /app

