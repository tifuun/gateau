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

## Testing

0. Make sure `podman` is installed and working.
1. Pull the tests images: `./podman/test-all.sh pull`
    - You can also build the images locally:
        `./podman/test-all.sh build`
2. Run the tests: `./podman/test-all.sh test`
3. Check results: the results of `pip install` are saved in
    `./podman/output/*`. They must all be zero.
    Otherwise, something went wrong.



