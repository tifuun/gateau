"""!
@file File containing some commonly encountered scanning patterns.
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

    az = np.ones(times.size) * az0
    el = np.ones(times.size) * el0
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

    az = mods * throw + np.ones(times.size) * az0
    el = np.ones(times.size) * el0
    return az, el


