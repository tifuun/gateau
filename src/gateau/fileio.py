import numpy as np
import os
import struct

def unpack_output(path: str, 
                  path_spaxel: str,
                  chunk_idx: int) -> dict[str, np.ndarray]:
    """!
    Process binary files written with results from simulation into output jsons for further processing.
    """
    ntimes = 0
    nfreq = 0

    out = {
            "signal"    : [],
            "az"        : [],
            "el"        : [],
            "time"      : [],
            }

    with open(os.path.join(path, f"{chunk_idx}time.out"), 'rb') as fh:
        data = fh.read()
        ntimes = len(data)//4
        data_flag = struct.unpack(f"@{ntimes}f", data)
        out["time"] = np.array(data_flag)
    
    with open(os.path.join(path_spaxel, f"{chunk_idx}az.out"), 'rb') as fh:
        data = fh.read()
        data_az = struct.unpack(f"@{ntimes}f", data)
        out["az"] = np.array(data_az)
    
    with open(os.path.join(path_spaxel, f"{chunk_idx}el.out"), 'rb') as fh:
        data = fh.read()
        data_el = struct.unpack(f"@{ntimes}f", data)
        out["el"] = np.array(data_el)
    
    with open(os.path.join(path_spaxel, f"{chunk_idx}signal.out"), 'rb') as fh:
        data = fh.read()
        nsig = len(data)//4
        data_l = struct.unpack(f"@{nsig}f", data)
        nfreq = nsig // ntimes
        out["signal"] = np.array(data_l).reshape((nfreq, ntimes)).T

    return out
