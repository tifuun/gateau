import numpy as np
import multiprocessing
from scipy.ndimage import generic_filter
import scipy.constants as scc
import os
from functools import partial
import warnings
from numpy.fft import fft2, fftshift, fftfreq
from scipy.interpolate import griddata

from typing import Tuple

import logging

from gateau.custom_logger import CustomLogger, parallel_iterator

logging.getLogger(__name__)
warnings.filterwarnings("ignore")

NCPU = multiprocessing.cpu_count()
FACTOR_PAD = 3

def convolve_source_cube_pool(args: Tuple[np.ndarray, np.ndarray, int], 
                              diameter_tel: float, 
                              az_arr: np.ndarray, 
                              el_arr: np.ndarray,
                              edge_taper: float,
                              source_cube_unit: str) -> np.ndarray:

    az_rad = az_arr / 180 * np.pi
    el_rad = el_arr / 180 * np.pi

    d_az = np.nanmean(np.diff(az_rad))
    d_el = np.nanmean(np.diff(el_rad))

    omega_pixel = d_az * d_el * np.cos(el_rad)[None,:]

    source_cube, f_src, thread_idx = args
    Rtel = diameter_tel / 2

    lam_arr = scc.c / f_src
    
    source_cube_convolved = np.zeros(source_cube.shape)

    for i, lam in enumerate(parallel_iterator(lam_arr, thread_idx)):
        cval = (np.nanmean(source_cube[:,0,i]) + np.nanmean(source_cube[:,-1,i]) + \
                np.nanmean(source_cube[0,:,i]) + np.nanmean(source_cube[-1,:,i]))
        
        ff = ff_from_aperture(az_arr, el_arr, lam, Rtel, edge_taper)
        omega_beam = np.nansum(omega_pixel * ff)

        norm = 1
        etendu = np.pi*Rtel**2

        if source_cube_unit == "I_nu":
            norm = 1#omega_pixel#np.nansum(ff)
            #etendu *= omega_pixel#lam**2

        func = partial(moving_sum, 
                       ff_pattern=ff,
                       norm=norm)

        if source_cube_unit == "F_nu_beam":
            source_cube_convolved[:,:,i] = etendu * source_cube[:,:,i]

        else:
            source_cube_convolved[:,:,i] = etendu * generic_filter(source_cube[:,:,i] * omega_pixel, 
                                                      func, 
                                                      size=ff.shape,
                                                      mode="constant",
                                                      cval=cval)
    return source_cube_convolved

def moving_sum(source_slice, ff_pattern, norm):
    return np.nansum(source_slice * ff_pattern.ravel()) / norm

def convolve_source_cube(source_cube: np.ndarray,
                         az_arr: np.ndarray,
                         el_arr: np.ndarray,
                         f_src: np.ndarray,
                         diameter_tel: float,
                         edge_taper: float = -10,
                         source_cube_unit: str = "I_nu",
                         num_threads: int = NCPU) -> np.ndarray:
    
    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()
    
    clog.info("\033[1;32m*** CONVOLVING SOURCE CUBE ***")
    
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
                           el_arr = el_arr,
                           edge_taper=edge_taper,
                           source_cube_unit=source_cube_unit)
    
    with multiprocessing.get_context("spawn").Pool(num_threads) as pool:
        out = np.concatenate(pool.map(func_to_pool, args), axis=-1)

    return out

def ff_from_aperture(az_arr, 
                     el_arr, 
                     lam, 
                     R,
                     edge_taper):

    az_c = np.nanmean(az_arr)
    el_c = np.nanmean(el_arr)

    nu = az_arr.size*FACTOR_PAD
    nv = el_arr.size*FACTOR_PAD

    Rk = R  / lam

    sigma = Rk / np.sqrt(2 * np.log(10**(-edge_taper/20)))

    u = np.linspace(-Rk, Rk, nu)*FACTOR_PAD
    v = np.linspace(-Rk, Rk, nv)*FACTOR_PAD
                                  
    du = u[1] - u[0]              
    dv = v[1] - v[0]              

    ugr, vgr = np.mgrid[-Rk*FACTOR_PAD:Rk*FACTOR_PAD:nu*1j,
                        -Rk*FACTOR_PAD:Rk*FACTOR_PAD:nv*1j]

    mask_R = np.sqrt(ugr**2 + vgr**2) < Rk
    #aper_power = np.exp(-0.5 * (ugr**2 + vgr**2) / sigma**2) * mask_R
    aper_power = np.exp(-0.5 * (ugr**2 + vgr**2) / sigma**2) * mask_R


    ff_pattern = np.absolute(fftshift(fft2(aper_power)))**2
    ff_pattern /= np.nanmax(ff_pattern)

    az_fft = np.arcsin(fftshift(fftfreq(nu, d=du))) * 180 / np.pi + az_c
    el_fft = np.arcsin(fftshift(fftfreq(nv, d=dv))) * 180 / np.pi + el_c


    az_fft_gr, el_fft_gr = np.mgrid[az_fft[0]:az_fft[-1]:nu*1j, 
                                    el_fft[0]:el_fft[-1]:nv*1j]
    
    az_gr, el_gr = np.mgrid[az_arr[0]:az_arr[-1]:az_arr.size*1j, 
                            el_arr[0]:el_arr[-1]:el_arr.size*1j]

    ff_pattern_interp = griddata((az_fft_gr.ravel(), el_fft_gr.ravel()), ff_pattern.ravel(), (az_gr, el_gr), method="linear")

    return ff_pattern_interp
