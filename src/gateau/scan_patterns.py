"""!
@file scan_patterns.py
@brief File containing some commonly encountered scanning patterns.
"""

from typing import Union
import numpy as np

def stare(times: np.ndarray, 
          az0: Union[float, np.ndarray], 
          el0: Union[float, np.ndarray]) -> Union[np.ndarray, np.ndarray]:
    """!
    Get scanning pattern for a stare.

    @param times Numpy array containing times of scan.
    @param az0 Azimuth co-ordinate to stare at.
    @param el0 Elevation co-ordinate to stare at.

    @returns Array of azimuth and elevation co-ordinates for the observation.

    @ingroup public_api_scan_patterns
    """

    az, el = check_size(times, az0, el0)

    return az, el 

def chop(times: np.ndarray, 
         az0: Union[float, np.ndarray], 
         el0: Union[float, np.ndarray], 
         duty_cycle: float, 
         throw: float) -> Union[np.ndarray, np.ndarray]:
    """!
    Get scanning pattern for a sky chopper, alternating between two points on sky.
    Currently, only chopping in azimuth direction is supported.

    @param times Numpy array containing times of scan. Units: seconds.
    @param az0 Azimuth co-ordinate of the reference chop position. Units: deg.
    @param el0 Elevation co-ordinate to stare at. Units: deg.
    @param duty_cycle Frequency of a full revolution of the chopper blade. Units: Hz.
    @param throw Angular distance between reference chop and other chop. Units: deg.

    @returns Array of azimuth and elevation co-ordinates for the observation.

    @ingroup public_api_scan_patterns
    """
    
    n = np.floor(times * 2 * 2 * duty_cycle)
    mods = np.mod(n, 2)
    
    az0, el0 = check_size(times, az0, el0)

    az = mods * throw + az0
    el = el0
    return az, el

def daisy(times: np.ndarray,
          az0: Union[float, np.ndarray], 
          el0: Union[float, np.ndarray],
          r_petal: float,
          f_cycle: float,
          f_petal: float) -> Union[np.ndarray, np.ndarray]: 
    """!
    Get scanning pattern for a daisy scan.

    @param times Numpy array containing times of scan. Units: seconds.
    @param az0 Azimuth co-ordinate of the center of the daisy scan. Units: deg.
    @param el0 Elevation co-ordinate of the center of the daisy scan. Units: deg.
    @param r_petal Distance of center of scan to edge of the petal. Units: deg.
    @param f_cycle Frequency of a full cycle. Units: Hz.
    @param f_petal Frequency of a single petal. Units: Hz.

    @returns Array of azimuth and elevation co-ordinates for the observation.

    @ingroup public_api_scan_patterns
    """

    az0, el0 = check_size(times, az0, el0)

    az = r_petal * np.cos(2*np.pi * f_cycle * times) * np.sin(2*np.pi * f_petal * times) + az0
    el = r_petal * np.sin(2*np.pi * f_cycle * times) * np.sin(2*np.pi * f_petal * times) + el0
    
    return az, el 

def chopnod(times: np.ndarray,
            az0: Union[float, np.ndarray],
            el0: Union[float, np.ndarray],
            chop_duty_cycle: float,
            nod_duty_cycle: float,
            throw: float) -> Union[np.ndarray, np.ndarray]:
    """!
    Get scanning pattern for chopping and nodding.
    In this observation mode, the telescope observes 3 positions: one on-source and two off-source.
    It first rapidly chops between the on-source and one of the off-source position.
    Then it moves to alternating between on-source and the other off-source position.
    Currently, only chopping and nodding in azimuth direction is supported.

    @param times Numpy array containing times of scan. Units: seconds.
    @param az0 Azimuth co-ordinate of the reference chop position. Units: deg.
    @param el0 Elevation co-ordinate to stare at. Units: deg.
    @param chop_duty_cycle Frequency of a full revolution of the chopper blade. Units: Hz.
    @param nod_duty_cycle Frequency of a full nod cycle (both telescope positions). Units: Hz.
    @param throw Angular distance between reference chop and other chop. Units: deg.

    @returns Array of azimuth and elevation co-ordinates for the observation.

    @ingroup public_api_scan_patterns
    """
    # for even nods, nod_coding = 0, for odd nods, nod_coding = 1
    numbered_nods = np.floor(times * 2 * nod_duty_cycle)
    nod_coding = np.mod(numbered_nods, 2)

    # for even chops, chop_coding = 0, for odd chops, chop_coding = 1
    numbered_chops = np.floor(times * 2 * 2 * chop_duty_cycle)
    chop_coding = np.mod(numbered_chops, 2)

    az0, el0 = check_size(times, az0, el0)

    # on source: nod_coding = 0 and chop_coding = 0 or nod_coding = 1 and chop_coding = 1 --> az = az0
    # off source 1: nod_coding = 0 and chop_coding = 1 --> az = az0 - throw
    # off source 2: nod_coding = 1 and chop_coding = 0 --> az = az0 + throw
    az = az0 + (nod_coding - chop_coding) * throw
    el = el0
    return az, el

def check_size(times, az0, el0):
    """!
    Check size of az0 and el0 values.
    If az0 and/or el0 are float, resizes to times array.

    """
    az_ret = az0
    el_ret = el0
    
    if isinstance(az0, float) or isinstance(az0, int):
        az_ret *= np.ones(times.size)
    
    if isinstance(el0, float) or isinstance(el0, int):
        el_ret *= np.ones(times.size)

    return az_ret, el_ret
