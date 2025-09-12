[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10528021.svg)]
(https://doi.org/10.5281/zenodo.10528021)

Welcome to the `gateau` Github repository:
the GPU-Accelerated Time-dEpendent observAtion simUlator! 
This is the end-to-end simulator for TIFUUn observations,
and is currently still in progress.
For more info, please see
[the documentation pages](https://arend95.github.io/tiempo2/) (to be fixed).

## Installation

Gateau is available from [pypi](https://pypi.org/project/gateau/).
You can install it like any other python package:

```
pip install gateau
```

> [!WARNING]
> You must also install the
> [GNU Scientific Library](https://www.gnu.org/software/gsl/)
> (Often called `gsl` or `libgsl`) for Gateau to run properly.
> We will fix this in the future.


### Supported Platforms

We currently only support Linux with GNU libc
(known as `manylinux` in the Python world).
We do not ship wheels for other operating systems
or linuxes with other libc implementations.
If you want to get gateau working on one of these platforms,
see the [Compiling section](#manual-compiling) below
(and please let us know if you're interested in helping
us get gateau running on other platforms!).

## Manual Compiling

The general procedure for compiling Gateau
from this repo is like so:

1. Install Python, gcc, cmake, `gsl-devel` using your system package manager.
    - `gsl-devel` is usually called `libgsl-dev`
        on Debian-like distros.
1. (optional) create and activate venv
1. `pip install -e .`

Pip should automatically compile `libgateau.so`
(which is placed under `src/gateau/libgateau.so`
when installing in pip's Editable mode)
using cmake.
After compilation is done,
you can verify that things work using `unittest`.
Note that some tests will always fail if you don't
have a GPU that gateau can use:

1. `python -m unittest`


## For developers: using the `test-all.sh` script

Building, testing, and packaging Gateau requires a very hairy toolchain
and some rather counterintuitive command line incantations.
To make our lives easier, we have written a script which automates
all of this using Podman.
Here's how to use it:

1. Install and set up `podman` using your system's package manager
1. Set up `podman` for use with CUDA.
    Read [NVIDIA's docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
    or the [STRATAL SYSTEMS Wiki page](https://github.com/stratal-systems/wiki/blob/main/wiki/void-linux-cuda.md)
    for instructions on how to do this.
    - If you do not have a GPU,
        you can skip this step and
        instead run the script with
        `GATEAU_DISABLE_GPU=1`.
        Things that require a GPU will of course fail,
        but things like compiling and packaging
        will still work.
1. Run `./podman/test-all.sh pull` to download all of the containers
    needed for testing, building, and packaging Gateau.
    - Alternatively, you can use `./podman/test-all.sh build`
        to build the containers locally

What follows below is a brief cookbook of things you can do
with the `test-all.sh` script.
If you want a complete documentation of what
it does and how,
read the
[comment block at the start of the script itself](./podman/test-all.sh)

### Run the test suite

```
./podman/test-all.sh test11
```
Will run the full test suite under a cuda11 container
with multiple versions of Python.
The detailed output will be placed in `/podman/output` directory.

If you do not have a GPU, you can still run the tests like this:

```
GATEAU_DISABLE_GPU=1 ./podman/test-all.sh test11
```

Mosts tests will fail if there is no GPU.

### Build a wheel

```
./podman/test-all.sh wheel
```

This will build wheel and place it into the `dist` subdirectory.

### Upload to pypi

This will upload all of the wheels and source archives under `dist`
onto PyPI.
You will need to pass your PyPI token using the `TWINE_PASSWORD`
environment variable.
You can make a token in the pypi webui.

```
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-xxxxxxx --verbose --skip-existing --non-interactive dist/*
```

### Test the PyPI package

This will download the latest Gateau from PyPI,
install it, and run the test suite.
As with the `test11` subcommand,
the detailed output will be available under `podman/output`.

```
./podman/test-all.sh test-pypi
```

### Upload/test on TestPyPI

There exists a "playground" version of PyPI
at <https://test.pypi.org/>.
You can use it to play around with the Gateau packing process.
Instead of using the `pypi` and `test-pypi`
subcommands,
use the `tpypi` and `test-tpypi` subcommands
to target TestPyPI instead.

## For developers: building and publishing wheels

The wheel building and publishing process is automated using the
`test-all.sh` script,
please read
[the section on using it](#for-developers-using-the-test-allsh-script).
Otherwise, you can follow the manual instructions below:

To build wheels for gateau and upload to pypi
you will need all of the requirements for compiling
Gateau
(see the [Compiling section](#manual-compiling)),
plus the `twine` and `build` Python packages.

1. Make sure the package version specified in `pyproject.toml`
    is the version number that you want to publish.
1. Build wheel: run `python -m build` from the root of the repo.
    The results (`.whl` wheel and `.tag.gz` source archive)
    are placed into the `dist` subdirectory.
1. Fix name of wheel:
    ~~For some godforsaken reason~~ `build` generates wheel files
    that are named like `gateau-0.1.0-cp313-cp313-linux_x86_64.whl`,
    which PyPI WILL NOT ACCEPT.
    You need to change the part that says `linux_x86_64`
    into `manylinux_<majver>_<minver>`, where
    `<majver>` and `<minver>` are the version numbers of GNU libc.
    So for example if `build` generates a file named
    `gateau-0.1.0-cp313-cp313-linux_x86_64.whl`,
    and `ldd --version` tells you that you have
    `ldd (Ubuntu GLIBC 2.31-0ubuntu9.12) 2.31`,
    then you need to rename the file to
    `gateau-0.1.0-cp313-cp313-manylinux_2_31_x86_64.whl`.
    Read [this PEP](https://peps.python.org/pep-0427/#file-name-convention)
    for more info.
1. Upload to pypi:
    There are many ways to authenticate to pypi, but the easiest is to
    create an access token in the webui
    and pass it to `twine` via the `TWINE_PASSWORD` environment variable:
    `TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-xxxxxxx --verbose --skip-existing --non-interactive dist/*`


## Misc. notes for various linux distros

### Void Linux

- no clue about getting cuda to work, but that
    is not needed to just compile
- `xbps-install -Syu gcc cmake gsl-devel`

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

## Other misc notes for developers

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

