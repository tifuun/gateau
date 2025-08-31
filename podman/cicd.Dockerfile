FROM docker.io/nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
#
# "Bad practice" this "code smell" that
# SHUT UP the real BaD PrAcTiCe here is nvidia using
# ancient crusty outdated garbage like Debian
# that is more focused on starting up
# a bloody conversation with the user
# than doing what its told
#
# Yes, debian is garbage, I said it. Bite me!

RUN \
	apt update -y && \
	apt install -y \
		libgsl-dev \
		software-properties-common \
		&& \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt update -y && \
	apt install -y \
		python3.13 python3.13-venv \
		&& \
	\
	python3.13 -m venv /venv3.13 && \
	\
	/venv3.13/bin/pip install ruff build twine && \
	ln -sf /venv3.13/bin/ruff /bin/ruff && \
	:

