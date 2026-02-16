"""!
@file
Generate or read filterbank matrix.
"""

import numpy as np


def lorentzian(f, f0, Ql):
    """Normalized Lorentzian (directional filter) centered at f0."""
    return 1 / (1 + 4 * Ql**2 * (f / f0 - 1) ** 2)


def lorentzian_with_second_order(f, f0, Ql, I=1):
    """Normalized Lorentzian (directional filter) centered at f0."""
    return 1 / (1 + 4 * Ql**2 * (f / f0 - 1) ** 2) + I / 2 / (1 + 4 * Ql**2 * (f / (2 * f0) - 1) ** 2)


def generate_filterbank(instrumentDict: dict[str, any], 
                        sourceDict: dict[str, any]) -> np.ndarray:
    """   
    Generate an 'idealized' filterbank that assumes perfect matching of the filters to the signal line.

    Args:
        f (array): frequencies over which to evaluate the filterbank.
        f0 (array): a list of resonance frequencies of the filters.
        Ql (single value, array): The loaded quality factor of the filters, given as a single value or a list of the same length as f0_list.
        kernel (function): the function describing the filter. Must take as arguments (f, f0, Ql).

    Returns:
        transfer (2d array): the transfer function for each filter. dimensions (len(f),len(f0)).

    """
    
    Ql          = instrumentDict.get("R")
    f0          = instrumentDict.get("f_ch_arr")
    f           = sourceDict.get("f_src")
    
    kernel = lorentzian
    if instrumentDict["sec_harmonic"]:
        kernel = lorentzian_with_second_order

    transfer = np.ones((len(f), len(f0)))
    if np.isscalar(Ql):
        Ql = np.full(len(f0), Ql)

    idx_sort = np.flip(np.argsort(f0))

    factor_resonator = 1
    if instrumentDict.get("resonator_type") == "half-wave":
        factor_resonator = 0.5

    for i_in_idx, i in enumerate(idx_sort):
        f0_i = f0[i]
        Ql_i = Ql[i]

        L0 = factor_resonator * kernel(f, f0_i, Ql_i)
        T = L0.copy()
        T_star = np.zeros(len(f))
        for j in idx_sort[:i_in_idx]:
            T_star += transfer[:, j]
        T *= 1 - T_star

        transfer[:, i] = T
    
    transfer = transfer.T

    if (cutoff := instrumentDict.get("cutoff")) is not None:
        transfer[:,f < cutoff] = 0
    
    return transfer

def generate_filterbank_independent(instrumentDict: dict[str, any], 
                                    sourceDict: dict[str, any]) -> np.ndarray:
    """!
    Generate a Lorentzian filterbank matrix from resolving power R.

    @param instrumentDict Instrument dictionary.
    @param sourceDict Source dictionary.

    @returns filterbank The Lorentzian filterbank.
    """

    Ql          = instrumentDict.get("R")
    f0          = instrumentDict.get("f_ch_arr")
    f           = sourceDict.get("f_src")
    
    if np.isscalar(Ql):
        Ql = np.full(len(f0), Ql)
    
    kernel = lorentzian
    if instrumentDict["sec_harmonic"]:
        kernel = lorentzian_with_second_order

    #lorentzian = (A * gamma**2 / ((f_src[None,:] - f_filt[:,None])**2 + gamma**2))**order
    transfer = lorentzian(f[None,:], f0[:,None], Ql[:,None])

    if (cutoff := instrumentDict.get("cutoff")) is not None:
        transfer[:,f < cutoff] = 0

    return transfer

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
        print(q)
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            print(r)
            x = spacing * 3 / 2 / np.sqrt(3) * q
            y = spacing * (r + q / 2)

            x_c.append(x)
            y_c.append(y)

    az = np.array(x_c)
    el = np.array(y_c)

    return az, el
