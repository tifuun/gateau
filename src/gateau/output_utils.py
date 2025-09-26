"""!
@file Utilities that can be called by users. 
"""
from importlib import resources as impresources

import os
import numpy as np
import csv

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

from gateau.fileio import get_num_chunks, unpack_output
from gateau import resources

def yield_output(path, spaxel = 0):
    """!
    Obtain iterator for a single spaxel output.
    The iterator yields chunks of observation.
    Use this function to obtain full access to all voxels inside a spaxel, on a chunk-to-chunk basis.

    @param path Path to folder containing output.
    @param spaxel Spaxel to yield from. Default: 0.
    
    @returns Iterator yielding separate observation chunks.
    """
    path_spaxel = os.path.join(path, str(spaxel))
    num_chunks_in_path = get_num_chunks(path_spaxel)

    for chunk_idx in range(num_chunks_in_path):
        yield unpack_output(path, path_spaxel, chunk_idx)

def extract_voxel_tod(path, voxel, spaxel = 0):
    """!
    Obtain full TOD for a single voxel inside a single spaxel.
    Use this function to obtain access to a full observation TOD, for a single voxel, for a single spaxel.
    """

    path_spaxel = os.path.join(path, str(spaxel))
    num_chunks_in_path = get_num_chunks(path_spaxel)

    for chunk_idx in range(num_chunks_in_path):
        out_full = unpack_output(path, path_spaxel, chunk_idx)
        if chunk_idx == 0:
            out = {
                    "signal"    : out_full["signal"][:,voxel],
                    "az"        : out_full["az"],  
                    "el"        : out_full["el"],
                    "time"      : out_full["time"]
                    }
        else:
            out["signal"] = np.concatenate((out["signal"],
                                            out_full["signal"][:,voxel]))
            out["az"] = np.concatenate((out["az"], out_full["az"]))
            out["el"] = np.concatenate((out["el"], out_full["el"]))
            out["time"] = np.concatenate((out["time"], out_full["time"]))

    return out

def average_over_filterbank(array_to_average: np.ndarray, 
                            filterbank: np.ndarray,
                            norm: bool = False) -> np.ndarray:
    if norm:
        div = np.nansum(filterbank, axis=1)
    else:
        div = 1
    sh_f = filterbank.shape
    assert array_to_average.size == sh_f[1]

    array_tiled = np.squeeze(array_to_average)[None,:] * filterbank
    return np.nansum(array_tiled, axis=1) / div












