"""!
@file
Generate or read filterbank matrix.
"""

import numpy as np

def generateFilterbankFromR(instrumentDict: dict[str, any], 
                            sourceDict: dict[str, any]) -> np.ndarray:
    """!
    Generate a Lorentzian filterbank matrix from resolving power R.

    @param instrumentDict Instrument dictionary.
    @param sourceDict Source dictionary.

    @returns filterbank The Lorentzian filterbank.
    """

    R           = instrumentDict.get("R")
    nf_ch       = instrumentDict.get("nf_ch")
    order       = instrumentDict.get("order")
    f_filt      = instrumentDict.get("f_ch_arr")
    f_src       = sourceDict.get("f_src")
    
    gamma       = f_filt[:,None] / (2*R)

    A = 1

    if instrumentDict["box_eq"]:
        A = 4 / np.pi
    lorentzian = (A * gamma**2 / ((f_src[None,:] - f_filt[:,None])**2 + gamma**2))**order

    if (cutoff := instrumentDict.get("cutoff")) is not None:
        lorentzian[:,f_src < cutoff] = 0

    return lorentzian 

def generate_fpa_pointings(instrumentDict: dict[str, any]) -> tuple[np.ndarray, 
                                                                   np.ndarray]:
    """!
    Generate a hexagonal far-field pointing model.

    @param instrumentDict Dictionary containing instrument specifications.

    @returns Tuple of arrays containing azimuth and elevation pointings per spaxel
    """

    x_c = []
    y_c = []

    radius = instrumentDict.get("radius")
    spacing = instrumentDict.get("spacing") 

    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            x = spacing * 3 / 2 / np.sqrt(3) * q
            y = spacing * (r + q / 2)

            x_c.append(x)
            y_c.append(y)

    az = np.array(x_c)
    el = np.array(y_c)

    return az, el
