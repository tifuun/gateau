FROM ubuntu:20.04

ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /setup

RUN \
	apt update && \
	apt install -y \
		git  `# Needed by mesonpy??` \
		git-annex  `# all git cmds will fail in annex repo w/o annex` \
		meson \
		python3.9 \
		python3.9-dev \
		python3.9-venv \
		wget \
		&& \
		:

# The standard gateau deps (sans cuda)
RUN apt-get -y install gcc g++ pkg-config libgsl-dev libhdf5-dev

# cuda
RUN \
	wget "${CUDA_KEYRING_URL}" && \
	dpkg -i cuda-keyring_1.1-1_all.deb && \
	apt-get update && \
	`# DO NOT INSTALL FULL TOOLCHAIN! ` && \
	`# It takes like 10gb of space and like 30min ` && \
	`# apt-get -y install cuda-toolkit-11-1 ` && \
	`# Install only the parts we need: ` && \
	apt-get -y install \
		cuda-nvcc-11-1 \
		libcufft-dev-11-1 \
		libcurand-dev-11-1 \
		&& \
	:

# Newer versions:
#ENV PATH=/usr/local/cuda/bin:${PATH}
# Older versions:
ENV PATH=/usr/local/cuda-11.1/bin:${PATH}

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


