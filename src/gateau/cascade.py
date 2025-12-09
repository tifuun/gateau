import numpy as np
import os
import yaml
import csv

from typing import Union

# constants
h = 6.62607004 * 10**-34  # Planck constant
k = 1.38064852 * 10**-23  # Boltzmann constant
e = 1.60217662 * 10**-19  # electron charge
c = 299792458.0  # velocity of light

TCMB = 2.725


def johnson_nyquist_psd(f_src: np.ndarray, 
                        T: float) -> np.ndarray:
    """!
    Johnson-Nyquist power spectral density.

    @param f_src Source frequencies. Units: Hz.
    @param T Temperature. Units: K.

    @returns Power spectral density. Units: W / Hz.
    """
    return h * f_src / np.expm1(h * f_src / (k * T))

def window_trans(
    f_src: np.ndarray,
    thickness: float,
    tandelta: float,
    neff: float,
    window_AR: bool,
    T_parasitic_refl: float,
    T_parasitic_refr: float) -> tuple[np.ndarray, 
                                      np.ndarray]:
    """!
    Calculates the window transmission.

    @param f Frequency. Units: Hz.
    @param thickness Thickness of the window/lens. Units: m.
    @param tandelta Loss tangent of window/lens dielectric.
    @param neff Refractive index of dielectric. Set to 1 to remove reflections. Units : None.
    @param window_AR Whether the window is supposed to be coated by Ar (True) or not (False).
    @param T_parasitic_refl Temperature of parasitic source seen in reflection, w.r.t. instrument.
    @param T_parasitic_refr Temperature of parasitic source seen in refraction..

    @returns List containing list of arrays of efficiencies as first element, and list of arrays of psd's seen by each stage as second element.
    """

    eta = []
    psd = []
    
    refl = ((1 - neff) / (1 + neff)) ** 2 * np.ones(f_src.size)
    psd_refl = johnson_nyquist_psd(f_src, T_parasitic_refl)
    psd_refr = johnson_nyquist_psd(f_src, T_parasitic_refr)

    if not window_AR:
        eta.append(1 - refl)
        psd.append(psd_refl)

    refr = np.exp(
        -thickness
        * 2
        * np.pi
        * neff
        * tandelta * f_src / c
    )

    eta.append(refr)
    psd.append(psd_refr)

    if not window_AR:
        eta.append(1 - refl)
        psd.append(psd_refl)

    return eta, psd

def eta_Al_ohmic(f_src: np.ndarray) -> np.ndarray: 
    """!
    Calculate Ohmic losses for aluminium over array of sky frequencies.
    
    @param f_src Numpy array containing source frequencies. Units: GHz
    
    @returns Array with eta values for Ohmic losses.
    """
    
    eta_Al_ohmic_850 = 0.9975  # Ohmic loss of an Al surface at 850 GHz.

    return 1.0 - (1.0 - eta_Al_ohmic_850) * np.sqrt(f_src / 850e9)

def sizer(eta: Union[np.ndarray, float], 
           f_src: np.ndarray, 
           f_eta: np.ndarray = None) -> np.ndarray:
    """!
    Resize efficiency term to new size.

    Used to vectorize or interpolate on efficiency terms.
    If efficiency is a scalar, an array is returned with the same size as f_src.
    If efficiency is an array with different size then f_src, an array containing frequencies at which eta is evaluated should also be passed.
    A 1D interpolation on f_src is then performed to evaluate eta on f_src.
    If efficiency is array with same size as f_src, it is returned as-is. 
    Responisibility to verify if the efficiencies are evaluated on the same frequencies as present in f_src is placed on the user.

    @param eta Efficiency term of some stage.
    @param f_src Numpy array containing source frequencies. Units: GHz
    @param f_eta Numpy array containing frequencies at which eta is evaluated.
                 Should only be passed when 1D interpolation is required and defaults to None.
    
    @returns Array with eta values, depending on input (see above).
    """

    if not hasattr(eta, "__len__"):
        return eta * np.ones(f_src.size)

    elif f_eta is not None:
        idx_sorted = np.argsort(f_eta)
        return np.interp(f_src, 
                         f_eta[idx_sorted], 
                         eta[idx_sorted])

    else:
        return eta

def read_from_folder(cascade_folder: str,
                     yaml_name: str = "cascade.yaml"
                     ) -> list[dict[any, any]]:
    """
    Generate a cascade list from a cascade folder.
    The folder should contain a YAML file containing the cascadelist.
    Any vector-valued efficiency terms should be provided inside the folder as a CSV file, with the first column containing frequencies at which the terms are evaluated and the second column containing the terms themselves.
    Then, the CSV can be referenced inside the YAML by passing the CSV name (including .csv) to the `eta_coup` field inside the YAML.

    Parameters
    ----------
    cascade_folder
        String containing path to folder containing cascade YAML and any related CSV files.

    yaml_name
        String containing the name of the YAML file containing the cascade.
        Defaults to 'cascade.yaml'.

    Returns
    ----------
    List containing the cascade.
    """

    assert(os.path.exists(cascade_folder))
    assert(os.path.exists(yaml_path := os.path.join(cascade_folder, 
                                       yaml_name)))

    with open(yaml_path) as stream:
        try:
            cascade_list = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for stage in cascade_list:
        for key, item in stage.items():
            if isinstance(item, str):
                if item.endswith(".csv"):
                    assert(os.path.exists(csv_path := os.path.join(cascade_folder,
                                                                   item)))
                    freq = []
                    vals = []
                    with open(csv_path, 'r', newline='') as csvfile:                
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            freq.append(float(row[0]))
                            vals.append(float(row[1]))

                    stage[key] = (np.array(freq), np.array(vals))

    return cascade_list

def save_cascade(cascade_list: list[dict[any, any]],
                 save_folder: str,
                 yaml_name: str = "cascade") -> None:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    name_index = 0

    cascade_to_write = []

    for stage in cascade_list:
        stage_dict = {}

        for key, item in stage.items():
            if isinstance(item, tuple):
                assert(item[0].size == item[1].size)
                np.savetxt(os.path.join(save_folder, 
                                        f"{name_index}.csv"), 
                           np.column_stack(item),
                           delimiter = ",")

                stage_dict[key] = f"{name_index}.csv"
                
                name_index += 1

            else:
                stage_dict[key] = item

        cascade_to_write.append(stage_dict)

    with open(os.path.join(save_folder, f"{yaml_name}.yaml"), 'w') as outfile:
              yaml.dump(cascade_to_write, outfile)

def get_cascade(cascade_list: list[dict[str, any]],
                f_src: np.ndarray) -> tuple[np.ndarray, 
                                            np.ndarray]:
    """!
    Calculate a cascade list, consisting of efficiency and psd per stage.

    @param cascade_list List containing, per element, the efficiency and coupling temperature of each stage in the cascade.
                        For reflective stages, the dictionary should contain either:
                            - A single eta and temperature
                            - A tuple with efficiencies and frequencies at which these are defines, and a temperature
        
                        For refractive stages, the dictionary should contain:
                            - thickness of dielectric in meters, 
                            - loss tangent, 
                            - effective refractive index, 
                            - whether to use AR coating, 
                            - temperature seen in reflection coming from the ISS, 
                            - and temperature seen in refraction.

    @param f_src Array with source frequencies. Units: GHz.
    
    @returns List with list of arrays containing efficiencies as first element, and list containing arrays of psd as second element. 
    """

    group_list = []
    
    for cascade in cascade_list:
        group_list.append(cascade.get("group"))
    
    group_list_red_uniq = list(dict.fromkeys([x for x in group_list if x is not None]))

    idx_group_list = [i if label is None else label for i, label in enumerate(group_list)]
    cascade_type_list = np.array([0 if x.get("eta_coup") is not None else 1 for x in cascade_list])
    idx_group_list_uniq = list(dict.fromkeys(idx_group_list))

    to_delete = []
    for group_label in group_list_red_uniq:
        index_list = [i for i, label in enumerate(idx_group_list) if label == group_label]
        idx_group_list_uniq[idx_group_list_uniq.index(group_label)] = index_list

        to_delete.extend(index_list[:-1])

    cascade_type_list_uniq = np.delete(cascade_type_list, to_delete)
    idx_group_list_uniq = [[x] if not hasattr(x, "__len__") else x for x in idx_group_list_uniq] 

    # Now get eta for groups
    all_eta = []
    all_psd = []
    eta_ap = np.ones(f_src.size)

    psd_cmb = johnson_nyquist_psd(f_src, TCMB)

    for casc_t, idx_group in zip(cascade_type_list_uniq, idx_group_list_uniq):
        eta_ap_flag = 0
        
        # This cascade group is reflective
        if casc_t == 0:
            eta_grouped = np.ones(f_src.size)

            if (T_casc := cascade_list[idx_group[0]].get("T_parasitic")) == "atmosphere":
                all_psd.append(-1*np.ones(f_src.size)) # Group couples to atmosphere: set all psd here to -1 and deal with it in CUDA backend.
                eta_ap_flag = 1
            else:
                all_psd.append(johnson_nyquist_psd(f_src, T_casc)) # Calculate psd for T_parasitic

            for idx_g in idx_group:
                eta_interp_flag = False
                
                if isinstance(eta := cascade_list[idx_g].get("eta_coup"), tuple): # eta_coup is a tuple; array of eta and frequencies
                    assert(len(eta) == 2)
                    
                    eta, F_eta = eta

                    assert(eta.size == F_eta.size)
                    
                    eta_interp_flag = True

                elif eta == "Ohmic-Al": # generate vector with eta of Aluminium
                    eta = eta_Al_ohmic(f_src) 

                if eta_interp_flag:
                    eta_grouped *= sizer(eta, f_src, F_eta)
                else:
                    eta_grouped *= sizer(eta, f_src)
            
            all_eta.append(eta_grouped)
            
            if eta_ap_flag:
                eta_ap *= eta_grouped
        
        if casc_t == 1:
            for idx_g in idx_group:
                casc = cascade_list[idx_g]
                etas, psds = window_trans(f_src,
                                          casc.get("thickness"), 
                                          casc.get("tandelta"), 
                                          casc.get("neff"), 
                                          casc.get("window_AR"), 
                                          casc.get("T_parasitic_refl"), 
                                          casc.get("T_parasitic_refr")) 
                all_eta.extend(etas)
                all_psd.extend(psds)
    
    return all_eta, all_psd, eta_ap, psd_cmb
