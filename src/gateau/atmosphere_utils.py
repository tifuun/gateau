"""!
@file atmosphere_utils.py
@brief Utilities for preparing atmosphere screens that can be called by users. 
    Also contains a function for generating atmospheric transmission curves.
"""

from importlib import resources as impresources

import os
import numpy as np
import csv
import multiprocessing
from functools import partial
from typing import Tuple

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad

import logging

from gateau.custom_logger import CustomLogger, parallel_iterator
from gateau import resources

logging.getLogger(__name__)
NCPU = multiprocessing.cpu_count()

def prep_atm_ARIS_pool(args: Tuple[np.ndarray,
                                   int], 
                       path_to_aris: str, 
                       radius_tel: float,
                       sigma: float) -> Tuple[float, float]:
    """!
    Function that does the work of 'prep_atm_ARIS'.
    It is not supposed to be used as-is, but rather to be passed to 'multiprocessing.Pool'.

    @param args Arguments over which parallelisation occurs.
        In this case, these are a list of ARIS screen names, and the thread index.
    @param path_to_aris Path to folder containing ARIS screens.
    @param radius_tel Radius of primary aperture of telescope, in meters.
    @param sigma Standard deviation of Gaussian power pattern of primary aperture.

    @returns Number of cells along x-direction, per screen.
    @returns Number of cells along y-direction, per screen.
    @returns Minimum PWV fluctuation for all screens.
    @returns Maximum PWV fluctuation for all screens.
    """
    
    truncate = radius_tel/sigma
    
    files, thread_idx = args

    min_PWV_arr = np.zeros(files.size)
    max_PWV_arr = np.zeros(files.size)
    
    for ii, file in enumerate(parallel_iterator(files, thread_idx)):
        file_split = file.split("-")

        file_idx = int(file_split[-1])
        
        subpath = os.path.join(path_to_aris, file)
        prepd_path = os.path.join(path_to_aris, "prepd")
        subchunk = np.loadtxt(subpath, delimiter=',')
        
        n_subx = np.unique(subchunk[:,0]).size
        n_suby = np.unique(subchunk[:,1]).size
       
        dEPL = subchunk[:,2].reshape((n_subx, n_suby))
        dPWV = (1./6.3003663 * dEPL*1e-6)*1e+3 #in mm
        
        dPWV_Gauss = gaussian_filter(dPWV, sigma, mode='mirror', truncate=truncate)

        min_PWV = np.nanmin(dPWV_Gauss)
        max_PWV = np.nanmax(dPWV_Gauss)

        dPWV_path = os.path.join(prepd_path, f"{file_idx}.datp")
        np.savetxt(dPWV_path, dPWV_Gauss)

        min_PWV_arr[ii] = min_PWV
        max_PWV_arr[ii] = max_PWV

    return n_subx, n_suby, np.nanmin(min_PWV_arr), np.nanmax(max_PWV_arr)

def prep_atm_ARIS(path_to_aris: str, 
                  radius_tel: float,
                  edge_taper: float = -10,
                  num_threads: int = NCPU) -> None:
    """!   
    Prepare ARIS atmospheric screens for usage in gateau.
    This function takes a path to dEPL ARIS screens, converts this to PWV using the Smith-Weintraub relation,
    and filters this with a truncated Gaussian corresponding to the power pattern of the primary aperture.
    The output screens are stored in a folder named '/prepd/', which is stored in the same folder as the ARIS screens.
    Run this function at least once per ARIS collection/telescope model combination.
        
    @ingroup public_API_atmosphere
    
    @param path_to_aris String containing the path to the folder containing the ARIS screens.
        The path can either be absolute or relative to your working directory.
    @param radius_tel Radius of primary aperture of telescope, in meters.
    @param edge_taper Power level of illumination pattern at rim of primary aperture, in decibel.
        Defaults to -10 dB.
    @param num_threads Number of CPU threads to use for the ARIS screen preparation.
        Defaults to the total number of threads on the CPU. 
    """

    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()

    clog.info("\033[1;32m*** PREPARING ARIS SCREENS ***")

    prepd_path = os.path.join(path_to_aris, "prepd")
    if not os.path.isdir(prepd_path):
        os.mkdir(prepd_path)

    screen_files = [f for f in os.listdir(path_to_aris) if os.path.isfile(os.path.join(path_to_aris, f))]
    chunks_screen_files = np.array_split(screen_files, num_threads)
    chunk_idxs = np.arange(0, num_threads)

    args = zip(chunks_screen_files, chunk_idxs)

    # We now calculate sigma of Gaussian.
    # Note that, since we want the sigma of the powwer pattern, we divide the edge taper by 10.
    # This gives us the sigma of the power pattern in the near field.
    sigma = radius_tel / np.sqrt(-2 * np.log(10**(edge_taper/10)))

    func_to_pool = partial(prep_atm_ARIS_pool, 
                           path_to_aris=path_to_aris,
                           radius_tel=radius_tel,
                           sigma=sigma)

    with multiprocessing.get_context("spawn").Pool(num_threads) as pool:
        out = pool.map(func_to_pool, args)

    min_PWV = np.nanmin(np.array([x[2] for x in out]))
    max_PWV = np.nanmax(np.array([x[3] for x in out]))

    with open(os.path.join(prepd_path, "atm_meta.datp"), "w") as metafile:
        metafile.write(f"{len(screen_files)} {out[0][0]} {out[0][1]} {min_PWV} {max_PWV}")

def get_eta_atm(f_src: np.ndarray,
                pwv0: float,
                el0: float) -> np.ndarray:
    """!
    Get atmospheric transmission curve for a range of frequencies.

    @param f_src Array of frequencies at which to evaluate atmospheric transmission, in Hz.
    @param pwv0 PWV at which to evaluate transmission, in mm.
    @param el0 Elevation at which to evaluate transmission in degrees.
    
    @returns Array of atmospheric transmission values.
    """
    
    with (impresources.files(resources) / 'eta_atm').open('r') as file:
        reader = csv.reader(file, delimiter=' ')
        pwv = []
        f_atm = []
        eta = []
        for i, row in enumerate(reader):
            if i == 0:
                pwv = np.array([float(x) for x in row])
                continue

            f_atm.append(float(row[0]))
            eta.append([float(x) for x in row[1:]])

        f_atm = np.array(f_atm) * 1e9
        eta = np.array(eta)

        eta_atm_interp = RectBivariateSpline(f_atm, 
                                             pwv,
                                             eta, 
                                             kx=1, ky=1)(f_src, 
                                                         pwv0,)

        return np.squeeze(eta_atm_interp) ** (1 / np.sin(el0 * np.pi / 180))

