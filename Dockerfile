FROM debian:bookworm

ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
ARG GSL_SIG_URL=https://mirror.clientvps.com/gnu/gsl/gsl-2.5.tar.gz.sig
ARG GSL_URL=wget https://mirror.clientvps.com/gnu/gsl/gsl-2.5.tar.gz

WORKDIR /setup

RUN \
	apt update && \
	apt install -y \
		autoconf \
		cmake \
		gcc \
		git \
		git-annex \
		gpg \
		hdf5-tools \
		libcurl4-openssl-dev `# needed?`\
		libhdf5-dev \
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
	wget "${CUDA_KEYRING_URL}" && \
	dpkg -i cuda-keyring_1.1-1_all.deb && \
	apt-get update && \
	apt-get -y install cuda-toolkit-12-3 && \
	echo "-----BEGIN PGP PUBLIC KEY BLOCK-----" > gsl_key.txt && \
	echo "Version: GnuPG v2.0.22 (GNU/Linux)" >> gsl_key.txt && \
	echo "" >> gsl_key.txt && \
	echo "mQENBFGmV38BCADRYBJRUS5FRv7LSlSY296SULeUmcNPp9enRBhN/0obENtGVJcP" >> gsl_key.txt && \
	echo "rspSylN4aQnCh7io3ESWDuKaz/1frqcpLdcPAqtN7qI+u522+DiBHAWnr0AdVLXP" >> gsl_key.txt && \
	echo "xllmHCqdzpgWwadGsAZ0H/u31XlkERhDNUnIFnw5HhsF2mJWX/yytusERcQbz/Ue" >> gsl_key.txt && \
	echo "MJMkwHW9n0htFCYkirV38nOmDJ843JmfMmregq2+E6MsDLXBc8L0kIPUIRzvm3sV" >> gsl_key.txt && \
	echo "I4WrI/SCKzl9262kOFeZXyTQ/5pFQUnnkBrbP39UlXIztSt9d1P3INAjv+e1ZZ7C" >> gsl_key.txt && \
	echo "0igHwndj+upJFROdfKO/UXYTMFgQ3zc6qbJ/ABEBAAG0IlBhdHJpY2sgQWxrZW4g" >> gsl_key.txt && \
	echo "PGFsa2VuQGNvbG9yYWRvLmVkdT6JATgEEwECACIFAlGmV38CGwMGCwkIBwMCBhUI" >> gsl_key.txt && \
	echo "AgkKCwQWAgMBAh4BAheAAAoJECRft0uuBbPpTvEH/0fiMqbKEsu66jNinMHdKQT5" >> gsl_key.txt && \
	echo "YN9Vq0IZi+PTO0PATlJ/s3FaLXZJ+v6Ag8NsrwSDH+Wrh86rVYOYyJrR7N0Mn0mr" >> gsl_key.txt && \
	echo "v6tBjjXx7n9MAzYZlizsvfQbm7Q2H5uJlM6AvfQRzSPG8nZGc3/+Xn6peefFwfpL" >> gsl_key.txt && \
	echo "nRJ/Xah1geqyiTNg3uInpzF7QHD6Rg9kX54xKF2s7g5PtgNNJxjKuM2xHnF4rot7" >> gsl_key.txt && \
	echo "UHE+S7dZ8qKmanlNwOhVXBI0EfDc3vK3D3JQmT6iI5pzE7huVKrGIxJXGS83zKLM" >> gsl_key.txt && \
	echo "urxUWzZ1hKhabxbkmryOK3ii2lkVMNdKcWPfHmQyjsVZpaVw9EGuQo1s4MN6Ac25" >> gsl_key.txt && \
	echo "AQ0EUaZXfwEIANRbLfjHVSZT0+IuRFRYNExWWOg/lY7/c7SD7Kqj5hFm6XWNXxRa" >> gsl_key.txt && \
	echo "IX8XNZI8mmRhrZZ4hX4qYk0EpVNtTKTxr1cG9Qk+FlKC9embqBL7Noj0ZEJTozlD" >> gsl_key.txt && \
	echo "t029xqW1G/trcqr2y0DKevfVzamhMgSHjmcEfscrcafYrYMxXASw/40Yiz/GWnDU" >> gsl_key.txt && \
	echo "EqEZb8XC9zSUCfuowpfbXxGGLFW5tFkW6hfgebePIUdx9RDdCu2Iuqf0v+hkZ6CR" >> gsl_key.txt && \
	echo "0vHp88aHdU/g6vRBrdwRZDd5wNOKvq1fMflvcsdf0RwOfuAwHWGcrAKs0nhqEYxj" >> gsl_key.txt && \
	echo "H1P8BLxL1xfPvGfANW2UWSce7mvKFEEY9y8AEQEAAYkBHwQYAQIACQUCUaZXfwIb" >> gsl_key.txt && \
	echo "DAAKCRAkX7dLrgWz6Ym2CACdH5EiDBPkDDjYa62r5gZ4Vel46jBSUcyni8Hq8wde" >> gsl_key.txt && \
	echo "YmN0FXKDBrq5G53aQp7bOyGHyU3u4Whsc0TnIbnXvhKTklxVOfuUKZQw+SnGQkMK" >> gsl_key.txt && \
	echo "apM30i5grtUKn5GJYFzX2GVhmCtIG7adtkvHiGXccWc9p6MFK4TRuRZ6Ut73i4l4" >> gsl_key.txt && \
	echo "CpZ0eHbJMNtbHTI+9VNzgvYcUWqzDPFNOyQ1275g+cMYTCaLE2W/MHLNzUjZe5hf" >> gsl_key.txt && \
	echo "3DFQjqea4ANCLyOh5IZNg5/v0KokCzz3Sruv4DQXxxWSF/jobifvFutjKqYDB4/c" >> gsl_key.txt && \
	echo "8hqk0PFuiiZFESCwD7Okg9ydxG1DFhK7zyk2JRGHbmNG" >> gsl_key.txt && \
	echo "=sAxL" >> gsl_key.txt && \
	echo "-----END PGP PUBLIC KEY BLOCK-----" >> gsl_key.txt && \
	gpg --import gsl_key.txt && \
	wget "${GSL_URL}" -O gsl-2.5.tar.gz && \
	wget "${GSL_SIG_URL}" -O gsl-2.5.tar.gz.sig && \
	gpg --verify gsl-2.5.tar.gz.sig && \
	tar xf ../vendor/gsl-2.5.tar.gz && \
	cd gsl-2.5 && \
	./autogen.sh && \
	sed -i -e '1i #!/bin/bash' ./configure && \
	./configure \
		--prefix=/usr \
		--enable-maintainer-mode \
		--disable-shared \
		--enable-static \
		&& \
	make -j$(nproc) && \
	make install && \
	python3.11 -m venv /venv && \
	. /venv/bin/activate && \
	pip install -r requirements.txt && \
	pip install build && \
	:



