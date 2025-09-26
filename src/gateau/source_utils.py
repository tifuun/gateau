import numpy as np
import multiprocessing
from scipy.ndimage import generic_filter
from scipy.special import j1
import scipy.constants as scc
import math
from functools import partial
import warnings
warnings.filterwarnings("ignore")

from typing import Tuple

from gateau.custom_logger import parallel_iterator

NCPU = multiprocessing.cpu_count()

def convolve_source_cube_pool(args: Tuple[np.ndarray, 
                                     np.ndarray, 
                                     int], 
                              diameter_tel: float, 
                              az_arr: np.ndarray, 
                              el_arr: np.ndarray) -> np.ndarray:
    

    source_cube, f_src, thread_idx = args
    Rtel = diameter_tel / 2

    d_az = az_arr[1] - az_arr[0]
    d_el = el_arr[1] - el_arr[0]

    k = 2 * np.pi * f_src / scc.c
    
    source_cube_convolved = np.zeros(source_cube.shape)

    for i, _k in enumerate(parallel_iterator(k, thread_idx)):
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

        _func = partial(airy, 
                        az_grid = az_grid_sub,
                        el_grid = el_grid_sub,
                        k = _k,
                        R = Rtel) 

        source_cube_convolved[:,:,i] = generic_filter(source_cube[:,:,i], 
                                                      _func, 
                                                      size=(n_fp_az, n_fp_el))
    return source_cube_convolved

def convolve_source_cube(source_cube: np.ndarray,
                         az_arr: np.ndarray,
                         el_arr: np.ndarray,
                         f_src: np.ndarray,
                         diameter_tel: float,
                         num_threads: int = NCPU)-> np.ndarray:
    
    print("\033[1;32m*** CONVOLVING SOURCE CUBE ***")
    
    chunks_source_cube = np.array_split(source_cube, 
                                        num_threads,
                                        axis=-1)
    chunks_f_src = np.array_split(f_src, 
                                  num_threads)
    chunk_idxs = np.arange(0, num_threads)

    args = zip(chunks_source_cube, 
               chunks_f_src, 
               chunk_idxs)

    func_to_pool = partial(convolve_source_cube_pool, 
                           diameter_tel=diameter_tel,
                           az_arr = az_arr, 
                           el_arr = el_arr)
    
    with multiprocessing.get_context("spawn").Pool(num_threads) as pool:
        out = np.concatenate(pool.map(func_to_pool, args), axis=-1)

    return out


def airy(source_slice, az_grid, el_grid, k, R):
    theta = np.radians(np.sqrt((az_grid)**2 + (el_grid)**2))
    airy = np.nan_to_num((2 * j1(k * R * np.sin(theta)) / (k * R * np.sin(theta)))**2, nan=1)

    return np.nansum(source_slice * airy.ravel()) / np.nansum(airy)

