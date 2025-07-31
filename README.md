[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10528021.svg)]
(https://doi.org/10.5281/zenodo.10528021)

Welcome to the `gateau` Github repository:
the GPU-Accelerated Time-dEpendent observAtion simUlator! 
This is the end-to-end simulator for TIFUUn observations,
and is currently still in progress.
For more info, please see
[the documentation pages](https://arend95.github.io/tiempo2/) (to be fixed).

## Installation

System-wide requirements (get these from your package manager):
- `gcc`
- `cmake`
- `gsl-devel`

Example for Void Linux: `xbps-install -Syu gcc cmake gsl-devel`

Once you have installed the system-wide requirements,
you can `pip install -e .` this repo.
All Python dependencies will be downloaded by pip.

<!--

## Testing

We need to test gateau on multiple cuda versions
and multiple python versions.
To achieve this, we have a script that
sets up different envrionments inside Podman containers.

First, you need to acquire the cuda11 and cuda12 images we have
prepared for testing gateau.
These are based on official NVIDIA images, but with
extra dependencies installed to make gateau work.

Download the images from the STRATAL SYSTEMS Docker registry:
```
./podman/test-all.sh 
```

-->


