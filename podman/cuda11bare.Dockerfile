FROM docker.io/nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN \
	apt update -y && \
	apt install -y \
		software-properties-common \
		`#libgsl23` \
		&& \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt update -y && \
	apt install -y \
		python3.13 python3.13-venv \
		python3.12 python3.12-venv \
		python3.11 python3.11-venv \
		python3.10 python3.10-venv \
		python3.9 python3.9-venv \
		&& \
	\
	python3.9 -m venv /venv3.9 && \
	python3.10 -m venv /venv3.10 && \
	python3.11 -m venv /venv3.11 && \
	python3.12 -m venv /venv3.12 && \
	python3.13 -m venv /venv3.13 && \
	:

