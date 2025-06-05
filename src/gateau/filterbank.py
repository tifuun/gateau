"""!
@file
Generate or read filterbank matrix.
"""

import numpy as np

def generateFilterbankFromR(instrumentDict, sourceDict):
    """!
    Generate a Lorentzian filterbank matrix from resolving power R.

    @param instrumentDict Instrument dictionary.

    @returns filterbank The Lorentzian filterbank.
    """

    R           = instrumentDict.get("R")
    nf_ch       = instrumentDict.get("nf_ch")
    order       = instrumentDict.get("order")
    f_filt      = instrumentDict.get("f_ch_arr")
    f_src       = sourceDict["f_src"]
    
    gamma       = f_filt[:,None] / (2*R)

    A = 1

    if instrumentDict["box_eq"]:
        A = 4 / np.pi

    return (A * gamma**2 / ((f_src[None,:] - f_filt[:,None])**2 + gamma**2))**order
