"""!
@file atmosphere_utils.py
@brief Utilities that can be called by users. 
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
from tqdm import tqdm

import logging

logging.getLogger(__name__)
from gateau.custom_logger import CustomLogger, parallel_iterator
from gateau import resources

NCPU = multiprocessing.cpu_count()

def prep_atm_ARIS_pool(args: Tuple[np.ndarray,
                                   int], 
                       path_to_aris: str, 
                       diameter_tel: float,
                       num_screens: int) -> None:
    Rtel = diameter_tel / 2
    std = Rtel/np.sqrt( 2.*np.log(10.) )
    truncate = Rtel/std
    
    files, thread_idx = args
    
    for file in parallel_iterator(files, thread_idx):
    
        file_split = file.split("-")

        file_idx = int(file_split[-1])
        
        subpath = os.path.join(path_to_aris, file)
        prepd_path = os.path.join(path_to_aris, "prepd")
        subchunk = np.loadtxt(subpath, delimiter=',')
        
        n_subx = np.unique(subchunk[:,0]).size
        n_suby = np.unique(subchunk[:,1]).size
       
        if not subchunk[0,0]:
            with open(os.path.join(prepd_path, "atm_meta.datp"), "w") as metafile:
                metafile.write(f"{num_screens} {n_subx} {n_suby}")

        dEPL = subchunk[:,2].reshape((n_subx, n_suby))
        dPWV = (1./6.3003663 * dEPL*1e-6)*1e+3 #in mm
        
        dPWV_Gauss = gaussian_filter(dPWV, std, mode='mirror', truncate=truncate)

        dPWV_path = os.path.join(prepd_path, f"{file_idx}.datp")
        np.savetxt(dPWV_path, dPWV_Gauss)

def prep_atm_ARIS(path_to_aris, diameter_tel, num_threads = NCPU):
    """!   
    Prepare ARIS atmospheric screens for usage in gateau.
    This function works in the following way:
        1. Load an ARIS subchunk, stored under the 'path' key in atmosphereDict.
           This requires the 'filename' key to include the extension of the file.
           This subchunk is a normal output artefact from ARIS, and therefore does not require further processing after being produced by ARIS.
        2. Remove ARIS metadata
        3. Convolve with 2D Gaussian
        4. Store ARIS data in subfolder (/prepd/) with same name.
    Run this function ONCE per ARIS collection/telescope site.
    
    @param atmosphereDict Dictionary containing atmosphere parameters.
    @param telescopeDict Dictionary containing telescope parameters.
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

    func_to_pool = partial(prep_atm_ARIS_pool, 
                           path_to_aris=path_to_aris,
                           diameter_tel=diameter_tel,
                           num_screens=len(screen_files))

    with multiprocessing.get_context("spawn").Pool(num_threads) as pool:
        pool.map(func_to_pool, args)

def get_eta_atm(f_src: np.ndarray,
                pwv0: float,
                el0: float) -> np.ndarray:
    
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
