# Info for maintainers

Info for maintainers is available in the docs (
[html](https://tifuun.github.io/gateau/maintainers.html),
[source](./doxy/maintainers.dox)
)

## NEW INFO

## Prepare dependencies

In order to build Gateau,
the following dependencies are required:

- [hdf5](https://github.com/HDFGroup/hdf5)
- [gsl](https://www.gnu.org/software/gsl/)

These can likely be installed from your package manager.

## Prepare CUDA toolchain

CUDA is required to build and run Gateau.
This can be installed from
[NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
**IMPORTANT**: Once installed, make sure `nvcc` is in your `$PATH`.

## Build 

Gateau uses [meson](https://mesonbuild.com/) as its build system,
You can likely install meson from your package manager.

Once the dependencies, meson, and CUDA have been installed,
you can install Gateau in development mode like this:

```
pip install -e .
```

Meson integrates tightly with the python packaging system.
You do not need to recompile Gateau every time you make changes;
recompilation will be triggered automatically every time `import gateau`
is invoked.

If you would still like to manually trigger recompilation,
it can be done as follows:

```
meson compile -C build
```

`build` is a directory in the root of the repo created by meson
to store temporary build files.
It is safe to remove.

## Building wheels

Building wheels is diffirent from building for development in two
different ways:

### Static libraries

You must use static versions of hdf5 and libgsl
so that they can be linked statically into `libgateau.so`.
This is standard practice for Python wheels.

Your distribution *may* ship static versions of both hdf5
and libgsl (i.e. `.a` files), either in the same package
that provides the dynamic versions (i.e. `.so` files),
or in a separate `*-static` package.

In case your distribution does not ship a static
hdf5 or gsl, you will need to build them from source.
We also offer a pre-built static libgsl package for debian
[here](https://github.com/stratal-systems/debian-packages/releases/tag/vtest2).

### Use `build` instead of `pip install -e`

In order to produce wheels that can be uploaded to PyPI,
use the `build` python module:

```
pip install build
python -m build
```

This will produce wheels under `dist`.

#### git integration

When using meson-python (as gateau does),
`build` will compile the latest *committed* version
of the project.
So, before running `build`,
either commit your changes,
or use `build` with a persistent builddir as detailed
in the
[meson-python docs](https://mesonbuild.com/meson-python/how-to-guides/config-settings.html).

#### Wheel names

By default, `build` will produce a wheel named

```
gateau-0.1.6-py3-none-any.whl
```

This is wrong. The `py3` implies that gateau is compatible with ALL
verisions and implementations of python3,
which it is not (at least cpython 3.9 is required),
and the `any` implies that gateau will work on any CPU architecture
(we only support `x86_64`).
Change this by simply renaming the wheel to:

```
gateau-0.1.6-cp39.cp310.cp311.cp312.cp313-none-none-manylinux_2_36_x86_64.whl
```

Replace `2_36` with the major and minor GLIBC version on the system
you used to compile Gateau,
or use the script provided under `scripts/wheelrename.sh`
to do this automatically.

## Building wheels using Docker

We provide a dockerfile that can be used to create a container image
with all of the relevant dependencies that are needed to build gateau.
Docker, rootless docker, and podman should all work with it.

Before proceeding, keep in mind that building the image
will take around half an hour and ~10GB of storage.


To build the image:

```
docker build . --tag gateau-builder
```

Then, to build gateau using the image:
```
docker run \
    --rm \
    --net none \
    -v .:/app \
    gateau-builder \
    sh -c '/venv/bin/python -m build --skip-dependency-check --no-isolation -Cbuild-dir=build.docker && ./scripts/wheelrename.sh'
```

The results will be available in the `dist` directory,
same as when using the `build` module natively.

The `--skip-dependency-check` and `--no-isolation` flags instruct `build`
to use the Python dependencies that were captured when the container was built,
effectively allowing Gateau to be built completely offline
(see the `--net none` flag).
If you want to use newer versions of existing dependencies
or new dependencies,
remove the `--skip-dependency-check`, `--no-isolation`, and `--net none`
flags;
or simply rebuild the image.

The `-Cbuild-dir=build.docker` flag tells meson to use the directory
named `build.docker` in the root of the repo as a cache for build files.
This allows for faster builds across container runs
without conflicting with the default `build` directory
that you may have created by building gateau on the host.

Note that we run the `wheelrename.sh` script inside the container as well
in order to capture the glibc version inside the container,
not on your host.

## Prebuilt container images

We have made a `docker save` archive of the gateau builder image
so that Gateau can still be built in case any of its dependencies
disappear from the internet.
However, we cannot share the container image publicly due
to the conflicting licenses of GSL and CUDA software
contained therein.
We can share it privately with other Gateau developers.



