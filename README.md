[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10528021.svg)]
(https://doi.org/10.5281/zenodo.10528021)

Welcome to the `gateau` Github repository:
the GPU-Accelerated Time-dEpendent observAtion simUlator! 
This is the end-to-end simulator for TIFUUn observations,
and is currently still in progress.
For more info, please see
[the documentation pages](https://arend95.github.io/tiempo2/) (to be fixed).

## Building and testing

### General procedure

1. Install Python, gcc, cmake, `gsl-devel`
1. (optional) create and activate venv
1. `pip install -e .`
1. `python -m unittest`
1. TODO gtest

What follows is notes and examples for various distros.
Tip: take a look at the dockerfile under `podman` for more info.

### Oracle Linux 9

- Available pythons: 3.9, 3.11, 3.12
- Extra repos need to be added for googletest:

```
dnf install -y oracle-epel-release-el9
dnf config-manager --set-enabled ol9_developer_EPEL
dnf install -y gtest gtest-devel gsl-devel python3.11

python3.11 -m venv /venv
source /venv/bin/activate

pip install -e .
python -m unittest
TODO gtest
```

### Ubuntu 20.04

- `deadsnakes` repo needed for non-ancient Pythons
    - contains almost all releases of python that you might
        ever want (3.9, 3.10, 3.11, 3.12, 3.13)
    - `venv` shipped separately,
        so you need to install both
        `pythonX.XX` and `pythonX.XX-venv`
- `gsl-devel` is called `libgsl-dev`
- TODO gtest

```
apt install -y libgsl-dev software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update -y
apt install python3.11 python3.11-venv

python3.11 -m venv /venv
source /venv/bin/activate

pip install -e .
python -m unittest
TODO gtest
```


## Installation

System-wide requirements (get these from your package manager):
- `gcc`
- `cmake`
- `gsl-devel`

Example for Void Linux: `xbps-install -Syu gcc cmake gsl-devel`

Once you have installed the system-wide requirements,
you can `pip install -e .` this repo.
All Python dependencies will be downloaded by pip.

## Testing

0. Make sure `podman` is installed and working.
1. Pull the tests images: `./podman/test-all.sh pull`
    - You can also build the images locally:
        `./podman/test-all.sh build`
2. Run the tests: `./podman/test-all.sh test`
3. Check results: the results of `pip install` are saved in
    `./podman/output/*`. They must all be zero.
    Otherwise, something went wrong.


## Misc notes

### Changing the signature of `run_gateau`

If you want to add/remove arguments of `run_gateau` there's
four places you need to change it:

- `InterfaceCUDA.h`
- `SimKernels.cu`
- `bindings.py` -- the Python function itself
- `bindings.py` -- the call to the C++ code


## License

Gateau is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, version 3 of the License only.

Gateau is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along
with Gateau. If not, see https://www.gnu.org/licenses/.

Note: Previous versions of gateau were released under different licenses. If the
current license does not meet your needs, you may consider using an earlier
version under the terms of its original license. You can find these versions by
browsing the commit history.

---

Copyright (c) 2025, maybetree, Arend Moerman.

