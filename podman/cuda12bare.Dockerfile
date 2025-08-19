FROM docker.io/nvidia/cuda:12.9.1-cudnn-devel-oraclelinux9 

RUN \
	dnf install python3.12 && \
	python3.12 -m venv /venv3.12 && \
	:

