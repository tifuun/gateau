[![DOI](https://zenodo.org/badge/988910511.svg)](https://doi.org/10.5281/zenodo.17183878)

Welcome to the `gateau` Github repository:
the GPU-Accelerated Time-dEpendent observAtion simUlator! 
This is the end-to-end simulator for TIFUUn observations,
and is currently still in progress.
For more info, please see
[the documentation pages](https://arend95.github.io/tiempo2/) (to be fixed).

## Installation

### Installing CUDA
CUDA is an API, developed by NVIDIA, to harness the computing power of a graphics processing unit (GPU) for general purpose computing. 
It provides access to the GPU's instruction set through common high-level programming languages, such as C and C++.

`gateau` uses CUDA for calculations, and hence must be installed in order to use `gateau`.
It can be [installed from source](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) or using a [package manager](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux).
`gateau` has been tested on both CUDA version 11 and 12, so please stick to these versions. 
Error-free performance on other versions is NOT guaranteed.

Even though `gateau` was exclusively developed on a [GTX 1650 Mobile](https://www.techpowerup.com/gpu-specs/geforce-gtx-1650-mobile.c3367), which is nowhere near impressive by today's standards, it is probably a good idea to use `gateau` on GPU's that meet or exceed the specs of this particular card.

### Installing gateau

Gateau is available from [pypi](https://pypi.org/project/gateau/).
You can install it like any other python package:

```
pip install gateau
```

The installation can be verified by opening a terminal, starting the python interpreter (make sure you are in the environment where `gateau` is installed), and running the following:
```
import gateau
gateau.selftest()
```

When installed correctly, the test should run without issues.


### Supported Platforms

We currently only support Linux with GNU libc
(known as `manylinux` in the Python world).
We do not ship wheels for other operating systems
or linuxes with other libc implementations.
If you want to get gateau working on one of these platforms,
have a read of [MAINTAINERS.md](./MAINTAINERS.md)
(and please let us know if you're interested in helping
us get gateau running on other platforms!).

## For maintainers and developers

For information on things like running tests
and making new releases of Gateau,
please consult [MAINTAINERS.md](./MAINTAINERS.md).
For contribution guidelines, see [CONTRIBUTING.md](./CONTRIBUTING.md)

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

