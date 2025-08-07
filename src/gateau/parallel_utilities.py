import numpy as np
from scipy.ndimage import generic_filter
from scipy.special import j1
import scipy.constants as scc
import math
from functools import partial

from typing import Tuple

def convolve_source_cube(args: Tuple[np.ndarray, 
                                     np.ndarray, 
                                     int], 
                         xargs: Tuple[float, 
                                      np.ndarray, 
                                      np.ndarray]) -> np.ndarray:
    """!
    Convolvbe a source cube with the diffraction-limited beam pattern of telescope.

    This function is not meant to be used directly, but passed to parallel_job_np instead.

    @param args Tuple of chunks of arguments, meant to be passed to each thread.
        In this case, args contains:
            f_arr: Array of source frequencies per thread.
            source_cube: Source slice per thread .
            thread_idx: Index of thread.
    @params xargs Extra arguments for function, the same for each thread.
        In this case, xargs contains:
            Dtel: Telescope primary diameter.
            az_grid: 2D numpy array specifying azimuth grid of source.
            el_grid: 2D numpy array specifying elevation grid of source.

    @returns Source cube convolved with telescope beam pattern, across all frequencies.
    """

    f_arr, source_cube, thread_idx = args
    Dtel, az_grid, el_grid = xargs
    Rtel = Dtel / 2

    d_az = az_grid[1,0] - az_grid[0,0]
    d_el = el_grid[0,1] - el_grid[0,0]

    k = 2 * np.pi * f_arr / scc.c
    
    source_cube_convolved = np.zeros(source_cube.shape)

    for i, _k in enumerate(_iterator(k, thread_idx)):
        lim = 100 * np.pi * 1.2 / _k / Rtel

        n_fp_az = math.ceil(lim / d_az)
        n_fp_el = math.ceil(lim / d_el)

        if n_fp_az % 2 == 0:
            n_fp_az += 1
        
        if n_fp_el % 2 == 0:
            n_fp_el += 1

        lim_az_sub = n_fp_az * d_az
        lim_el_sub = n_fp_el * d_el
        
        az_grid_sub, el_grid_sub = np.mgrid[-lim_az_sub:lim_az_sub:n_fp_az * 1j, 
                                            -lim_el_sub:lim_el_sub:n_fp_el * 1j]

        _func = partial(_AiryDisk, 
                        az_grid = az_grid_sub,
                        el_grid = el_grid_sub,
                        k = _k,
                        R = Rtel) 

        source_cube_convolved[:,:,i] = generic_filter(source_cube[:,:,i], 
                                                      _func, 
                                                      size=(n_fp_az, n_fp_el))
    return source_cube_convolved


def _AiryDisk(source_slice, az_grid, el_grid, k, R):
    theta = np.radians(np.sqrt((az_grid)**2 + (el_grid)**2))
    airy = np.nan_to_num((2 * j1(k * R * np.sin(theta)) / (k * R * np.sin(theta)))**2, nan=1)

    return np.nansum(source_slice * airy.ravel()) / np.nansum(airy)

def _iterator(x, idx_thread):
    try:
        from tqdm import tqdm
        return tqdm(x, 
                    ncols=100, 
                    total=x.size, 
                    colour="GREEN") if idx_thread == 0 else x
    except ImportError:
        print("No tqdm installed, so no fancy progress bar... :(")
        return x

