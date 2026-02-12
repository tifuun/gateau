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

    @ingroup scan_patterns
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

    @ingroup scan_patterns
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
          period_cycle: float,
          period_petal: float) -> Union[np.ndarray, np.ndarray]: 

    az0, el0 = check_size(times, az0, el0)

    az = r_petal * np.cos(2*np.pi * period_cycle * times) * np.sin(2*np.pi * period_petal * times) + az0
    el = r_petal * np.sin(2*np.pi * period_cycle * times) * np.sin(2*np.pi * period_petal * times) + el0
    
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
