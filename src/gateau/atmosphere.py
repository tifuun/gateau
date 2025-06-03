"""!
@file
This file handles all atmospheric screen generation. 

In particular, the EPL maps from ARIS are converted to Gaussian-smoothed PWV maps.
This script also contains functions that read atmospheric transmission curves, as function of frequency and PWV.
"""

import os
import numpy as np
import csv

from astropy.io import fits
from scipy.ndimage import gaussian_filter

def prepAtmospherePWV(atmosphereDict, telescopeDict, clog=None):
    """!   
    Prepare ARIS atmospheric screens for usage in TiEMPO2.
    This function works in the following way:
        1. Load an ARIS subchunk, stored under the 'path' key in atmosphereDict.
           This requires the 'filename' key to include the extension of the file.
        2. Remove ARIS metadata
        3. Convolve with 2D Gaussian
        4. Store ARIS data in subfolder (/prepd/) with same name.
    Run this function ONCE per ARIS collection/telescope site.
    """
    clog.info("\033[1;32m*** PREPARING ARIS SCREENS ***")
   
    filename = atmosphereDict.get("filename")
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

    test_l = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".datp"):
                continue
            clog.info(f"Preparing {file}...")
        
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
    
    if clog is not None:
        clog.info("\033[1;32m*** FINISHED PREPARING ARIS SCREENS ***")
    else:
        print(f"Finished preparing atmospheric screens.")

