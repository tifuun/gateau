"""!
@file Utilities that can be called by users. 
"""
from importlib import resources as impresources

import os
import numpy as np
import csv

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

from gateau.parallel import get_num_chunks 
from gateau.fileio import unpack_output
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

def prep_atm_ARIS(atmosphereDict, telescopeDict):
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

    print("\033[1;32m*** PREPARING ARIS SCREENS ***")
   
    path = atmosphereDict.get("path")
    pwv0 = atmosphereDict.get("PWV0")

    Rtel = telescopeDict.get("Dtel") / 2
    std = Rtel/np.sqrt( 2.*np.log(10.) )
    truncate = Rtel/std

    # Conversion from dEPL to dPWV from Smith-Weintraub relation.
    a = 6.3003663 

    prepd_path = os.path.join(path, "prepd")
    if not os.path.isdir(prepd_path):
        os.mkdir(prepd_path)

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".datp"):
                continue
            print(f"Preparing {file}...")
        
            file_split = file.split("-")

            file_idx = int(file_split[-1])
            
            subpath = os.path.join(path, file)
            subchunk = np.loadtxt(subpath, delimiter=',')
            
            n_subx = np.unique(subchunk[:,0]).size
            n_suby = np.unique(subchunk[:,1]).size
           
            if not subchunk[0,0]:
                with open(os.path.join(prepd_path, "atm_meta.datp"), "w") as metafile:
                    metafile.write(f"{len(files)} {n_subx} {n_suby}")

            dEPL = subchunk[:,2].reshape((n_subx, n_suby))
            PWV = pwv0 + (1./a * dEPL*1e-6)*1e+3 #in mm
            
            PWV_Gauss = gaussian_filter(PWV, std, mode='mirror', truncate=truncate)

            PWV_path = os.path.join(prepd_path, f"{file_idx}.datp")
            np.savetxt(PWV_path, PWV_Gauss)
        break
    
    print("Finished preparing atmospheric screens.")

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












