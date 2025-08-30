"""!
@file This file contains private functions for parallel execution of jobs on simulated data.
It is not meant to be used by users, only internally.
"""

import math
from multiprocessing import get_context
import numpy as np
import os
from collections.abc import Callable
from functools import partial

NFIELDS = 3

def get_num_chunks(path: str) -> int:
    return len(os.listdir(path)) // NFIELDS

def parallel_job_np(npfile: np.ndarray, 
                    num_threads: int, 
                    job: Callable, 
                    arr_par: np.ndarray, 
                    args_list: list[any], 
                    axis: int = None) -> np.ndarray:
    """!
    Perform a job on a numpy file in parallel.
    Note that this method does not calculate number of cores. This is purely user-specified.

    @param npfile Numpy file on which to apply a job in parallel.
    @param num_threads Number of CPU threads to use.
    @param job Function handle of function to apply to data.
    @param arr_par Array containing the axis of the numpy file over which to apply job in parallel.
    @param args_list List of extra arguments to pass to function.
    @param axis Axis of npfile along which to parallelise.

    @returns Output of job.
    """

    axis = -1 if axis is None else axis
    
    if arr_par.size < num_threads:
        num_threads = arr_par.size

    num_chunks = math.ceil(arr_par.size / num_threads)

    chunks_arr = np.array_split(arr_par, num_threads)
    chunks_np = np.array_split(npfile, num_threads, axis=axis)
    
    args = zip(chunks_arr, chunks_np, np.arange(0, num_threads))

    _func = partial(job, xargs = args_list)

    with get_context("spawn").Pool(num_threads) as pool:
        out = np.concatenate(pool.map(_func, args), axis=-1)

    return out

