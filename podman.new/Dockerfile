FROM docker.io/nvidia/cuda:12.9.1-cudnn-devel-oraclelinux9 AS cuda12-base
RUN \
	dnf install \
		gsl-devel \
		python3.9 python3.11 python3.12 \
		&& \
	python3.9 -m venv /venv3.9 && \
	python3.11 -m venv /venv3.11 && \
	python3.12 -m venv /venv3.12 && \
	:
