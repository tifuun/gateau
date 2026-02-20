FROM ubuntu:20.04

ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
ARG LIBGSL_DEB_URL=https://github.com/stratal-systems/ubuntu-20-04-packages/releases/download/v0-1/libgsl-static-pic_2.5.0_amd64.deb
ARG LIBHDF5_DEB_URL=https://github.com/stratal-systems/ubuntu-20-04-packages/releases/download/v0-1/libhdf5-static-pic_2.0.0_amd64.deb


ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /setup

RUN \
	apt update && \
	apt install -y \
		g++ \
		gcc \
		git  `# Needed by mesonpy??` \
		git-annex  `# all git cmds will fail in annex repo w/o annex` \
		meson \
		pkg-config \
		python3.9 \
		python3.9-dev \
		python3.9-venv \
		wget \
		&& \
		:

RUN \
	wget "${CUDA_KEYRING_URL}" && \
	dpkg -i cuda-keyring_1.1-1_all.deb && \
	apt-get update && \
	`# DO NOT INSTALL FULL TOOLCHAIN! ` && \
	`# It takes like 10gb of space and like 30min ` && \
	`# apt-get -y install cuda-toolkit-12-3 ` && \
	`# Install only the parts we need: ` && \
	apt-get -y install \
		cuda-nvcc-12-3 \
		libcufft-dev-12-3 \
		libcurand-dev-12-3 \
		&& \
	:

ENV PATH=/usr/local/cuda/bin:${PATH}

RUN \
	wget "${LIBGSL_DEB_URL}" && \
	wget "${LIBHDF5_DEB_URL}" && \
	dpkg -i *.deb && \
	rm -rf *.deb && \
	:

# These are all needed to build doxygen docs
# plus jupyter but that's handled by pip 
# (it's marked as an extra dep in pyproject.toml)
#
# Yeah, no, really. We pull in entirety of texlive
# just so doxygen can process the one single bibtex
# reference in peace.
# Whatever, storage is cheap.
RUN \
	apt install -y \
		doxygen \
		texlive-bibtex-extra \
		texlive-latex-base \
		&& \
	:

RUN \
	python3.9 -m venv /venv && \
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
	`#pip download -r requirements.txt -d pipcache --only-binary=:all: && ` \
	pip install -r requirements.txt && \
	pip install build twine && \
	:

WORKDIR /app


