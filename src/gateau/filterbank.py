"""!
@file
Generate or read filterbank matrix.
"""

import numpy as np

def generateFilterbankFromR(instrumentDict):
    """!
    Generate a Lorentzian filterbank matrix from resolving power R.

    @param instrumentDict Instrument dictionary.

    @returns filterbank The Lorentzian filterbank.
    """

    R           = instrumentDict.get("R")
    f0_ch       = instrumentDict.get("f0_ch")
    nf_ch       = instrumentDict.get("nf_ch")
    f0_src      = instrumentDict.get("f0_src")
    f1_src      = instrumentDict.get("f1_src")
    nf_src      = instrumentDict.get("nf_src")
    order       = instrumentDict.get("order")
    
    f_filt      = instrumentDict.get("f_ch_arr")

    A = 1

    if instrumentDict["box_eq"]:
        A = 4 / np.pi

    f_src = np.linspace(f0_src, f1_src, nf_src)

    instrumentDict["f_src_arr"] = f_src

    filterbank = np.zeros((nf_ch, nf_src))

    for j in range(nf_ch):
        _fj_ch = f_filt[j] 
        gamma = _fj_ch / (2 * R)
        filterbank[j,:] = (A * gamma**2 / ((f_src - _fj_ch)**2 + gamma**2))**order

    return filterbank
