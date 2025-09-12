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

## For maintainers and developers

For information on things like running tests
and making new releases of Gateau,
please consult [MAINTAINERS.md](./MAINTAINERS.md).
For contribution guidelines, see [CONTRIBUTING.md)[CONTRIBUTING.md]

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

