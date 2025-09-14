# Info for maintainers

This file contains all the hairy details that ideally should be
of no concern for users and casual contributors,
but are important for maybetree, Arend, and whatever poor souls
who will have to carry the burden of maintaining Gateau in the future.

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
./podman/test-all.sh staticwheel
```

This will build wheel and place it into the `dist` subdirectory.

If you are paranoid, you can run `./podman/test-all.sh checkstatic`
to verify that the wheel has libgsl linked statically
(more info in the [Libgsl](#libgsl) section).

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

## Libgsl

In Python Land, it is customary to statically
link dependencies into the wheel.
Our only dependency (besides C and C++ stdlibs,
which are halal to link dynamically afaik)
is the
[GNU Scientific Library](https://www.gnu.org/software/gsl/).
We link it statically into `libgateau.so`.
In order to accomplish this,
a static version of libgsl
itself is needed -- i.e. `libgsl.a` instead of `libgsl.so`.
Some distros provide a package for this,
but not Ubuntu (which is the base for the `gateau-cicd` container).
So we compile static libgsl manually.
This is done using the `scripts/build-gsl.sh` script,
which runs during the `staticwheel`
subcommand of the `test-all.sh` script.
That script has a little
non-trivial hackery to actually get libgsl to compile,
go take a look yourself,
I promise it's not that scary.

cmake is duumb. I have spent like an hour trying to figure out
how to tell it to link libgsl statically
and the best I have come up with is to just make
sure the container ONLY has the static libgsl (`libgsl.a`).
If the container has both static and dynamic (`libgsl.so`),
cmake defaults to the dynamic one no matter what I do.
This is the reason why the `build-gsl.sh` script
configures libgsl with `--distable-shared`,
and why the `staticwheel` subcommand has
that cheeky `rm -rf /usr/local/bin/libgsl.so*`,
and why there is the paranoid `checkstatic` subcommand as well.

Libgsl is licensed under GPLv3 which is the reason
we must use a GPL-compatible license for Gateau itself
if we want to distribute static wheels of it.

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

