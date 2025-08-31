FROM docker.io/nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
	apt update -y && \
	apt install -y \
		software-properties-common \
		libgsl23 \
		&& \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt update -y && \
	apt install -y \
		python3.13 python3.13-venv \
		&& \
	\
	python3.13 -m venv /venv3.13 && \
	:

