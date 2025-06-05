import numpy as np

from typing import List, Union, Dict, Tuple

# constants
h = 6.62607004 * 10**-34  # Planck constant
k = 1.38064852 * 10**-23  # Boltzmann constant
e = 1.60217662 * 10**-19  # electron charge
c = 299792458.0  # velocity of light
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]

eta_Al_ohmic_850 = 0.9975  # Ohmic loss of an Al surface at 850 GHz.

def johnson_nyquist_psd(F: ArrayLike, 
                        T: ArrayLike) -> ArrayLike:
    """!
    Johnson-Nyquist power spectral density.

    @param F Frequency. Units: Hz.
    @param T Temperature. Units: K.

    @returns Power Spectral Density. Units: W / Hz.
    """
    return h * F / np.expm1(h * F / (k * T))

def window_trans(
    F: ArrayLike,
    thickness: ArrayLike,
    tandelta: float,
    neffHDPE: float,
    window_AR: bool,
    T_parasitic_refl: float,
    T_parasitic_refr: float
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculates the window transmission.

    Parameters
    ----------
    F
        Frequency. Units: Hz.
    thickness
        Thickness of the HDPE window. Units: m.
    tandelta
        Loss tangent of window/lens dielectric.
    neffHDPE
        Refractive index of HDPE. Set to 1 to remove reflections. Units : None.
    window_AR
        Whether the window is supposed to be coated by Ar (True) or not (False).
    T_parasitic_refl
        Temperature of parasitic source seen in reflection, w.r.t. instrument.
    T_parasitic_refr
        Temperature of parasitic source seen in refraction..


    Returns
    -------
    List containing list of arrays of efficiencies as first element, and list of arrays of psd's seen by each stage.
    """

    eta = []
    psd = []
    
    HDPErefl = ((1 - neffHDPE) / (1 + neffHDPE)) ** 2 * np.ones(F.size)
    psd_refl = johnson_nyquist_psd(F, T_parasitic_refl)
    psd_refr = johnson_nyquist_psd(F, T_parasitic_refr)

    if window_AR == False:
        eta.append(1 - HDPErefl)
        psd.append(psd_refl)

    eta_HDPE = np.exp(
        -thickness
        * 2
        * np.pi
        * neffHDPE
        * (tandelta * F / c + (tandelta * F / c) ** 2)
    )

    eta.append(eta_HDPE)
    psd.append(psd_refr)

    if window_AR == False:
        eta.append(1 - HDPErefl)
        psd.append(psd_refl)

    return eta, psd

def eta_Al_ohmic(F_sky: np.ndarray) -> np.ndarray: 
    """Calculate Ohmic losses for aluminium over array of sky frequencies.
    
    Parameters
    ----------
    F_sky
        Numpy array containing sky frequencies. Units: GHz
    
    Returns
    ----------
    Array with eta values for Ohmic losses.
    """

    return 1.0 - (1.0 - eta_Al_ohmic_850) * np.sqrt(F_sky / 850e9)

def sizer(eta: Union[np.ndarray, float], 
           F_sky: np.ndarray, 
           F_eta: np.ndarray = None
) -> np.ndarray:
    """Resize efficiency term to new size.

    Used to vectorize or interpolate on efficiency terms.
    If efficiency is a scalar, an array is returned with the same size as F_sky.
    If efficiency is an array with different size then F_sky, an array containing frequencies at which eta is evaluated should also be passed.
    A 1D interpolation on F_sky is then performed to evaluate eta on F_sky.
    If efficiency is array with same size as F_sky, it is returned as-is. 
    Responisibility to verify if the efficiencies are evaluated on the same frequencies as present in F_sky is placed on the user.

    Parameters
    ----------
    eta
        Efficiency term.
    F_sky
        Numpy array containing sky frequencies. Units: GHz
    F_eta
        Numpy array containing frequencies at which eta is evaluated.
        Should only be passed when 1D interpolation is required and defaults to None.
    
    Returns
    ----------
    Array with eta values, depending on input (see above).
    """

    if not hasattr(eta, "__len__"):
        return eta * np.ones(F_sky.size)

    elif F_eta is not None:
        idx_sorted = np.argsort(F_eta)
        return np.interp(F_sky, 
                         F_eta[idx_sorted], 
                         eta[idx_sorted])

    else:
        return eta

def get_cascade(cascade_list: List[Dict[any, any]],
                F_sky: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate a cascade list, consisting of efficiency and psd per stage.

    Parameters
    ----------
    cascade_list
        List containing, per element, the efficiency and coupling temperature of each stage in the cascade.
        For reflective stages, the dictionary should contain either:
            - A single eta and temperature
            - A tuple with efficiencies and frequencies at which these are defines, and a temperature
        
        For refractive stages, the dictionary should contain:
            - thickness of dielectric in meters, loss tangent, effective refractive index, whether to use AR coating, temperature seen in reflection coming from the ISS, and temperature seen in refraction.

    F_sky
        Array with sky frequencies. Units: GHz.
    
    Returns
    ----------
    List with list of arrays containing efficiencies as first element, and list containing arrays of psd as second element. 
    """

    group_list = []
    
    for cascade in cascade_list:
        group_list.append(cascade.get("group"))
    
    group_list_red_uniq = list(dict.fromkeys([x for x in group_list if x is not None]))

    idx_group_list = [i if label == None else label for i, label in enumerate(group_list)]
    cascade_type_list = np.array([0 if x.get("eta_coup") is not None else 1 for x in cascade_list])
    idx_group_list_uniq = list(dict.fromkeys(idx_group_list))

    to_delete = []
    for group_label in group_list_red_uniq:
        index_list = [i for i, label in enumerate(idx_group_list) if label == group_label]
        idx_group_list_uniq[idx_group_list_uniq.index(group_label)] = index_list

        to_delete.extend(index_list[:-1])

    cascade_type_list_uniq = np.delete(cascade_type_list, to_delete)
    idx_group_list_uniq = [[x] if hasattr(x, "__len__") == False else x for x in idx_group_list_uniq] 

    # Now get eta for groups
    all_eta = []
    all_psd = []

    for casc_t, idx_group in zip(cascade_type_list_uniq, idx_group_list_uniq):
        # This cascade group is reflective
        if casc_t == 0:
            eta_grouped = np.ones(F_sky.size)

            if (T_casc := cascade_list[idx_group[0]].get("T_parasitic")) == "atmosphere":
                all_psd.append(-1*np.ones(F_sky.size)) # Group couples to atmosphere: set all psd here to -1 and deal with it in CUDA backend.
            else:
                all_psd.append(johnson_nyquist_psd(F_sky, T_casc)) # Calculate psd for T_parasitic

            for idx_g in idx_group:
                eta_interp_flag = False
                
                if isinstance(eta := cascade_list[idx_g].get("eta_coup"), tuple): # eta_coup is a tuple; array of eta and frequencies
                    assert(len(eta) == 2)
                    
                    eta, F_eta = eta

                    assert(eta.size == F_eta.size)
                    
                    eta_interp_flag = True

                elif eta == "Ohmic-Al": # generate vector with eta of Aluminium
                    eta = eta_Al_ohmic(F_sky) 

                if eta_interp_flag:
                    eta_grouped *= sizer(eta, F_sky, F_eta)
                else:
                    eta_grouped *= sizer(eta, F_sky)
            
            all_eta.append(eta_grouped)
        
        if casc_t == 1:
            for idx_g in idx_group:
                casc = cascade_list[idx_g]
                etas, psds = window_trans(F_sky,
                                          casc.get("thickness"), 
                                          casc.get("tandelta"), 
                                          casc.get("neff"), 
                                          casc.get("window_AR"), 
                                          casc.get("T_parasitic_refl"), 
                                          casc.get("T_parasitic_refr")) 
                all_eta.extend(etas)
                all_psd.extend(psds)
    
    return all_eta, all_psd
