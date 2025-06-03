import numpy as np
import matplotlib.pyplot as pt
import os
import struct

_type2fmt = {
        "f"     : 4,
        "d"     : 8,
        }

def unpack_output(path, chunk_idx):
    """!
    Process binary files written with results from simulation into output jsons for further processing.
    """
    ntimes = 0
    nfreq = 0
    dtype = "f"

    out = {
            "signal"    : [],
            "az"        : [],
            "el"        : [],
            "flag"      : [],
            }

    with open(os.path.join(path, f"{chunk_idx}flag.out"), 'rb') as fh:
        data = fh.read()
        ntimes = len(data)//4
        data_flag = struct.unpack(f"@{ntimes}i", data)
        out["flag"] = np.array(data_flag)
    
    with open(os.path.join(path, f"{chunk_idx}az.out"), 'rb') as fh:
        data = fh.read()
        
        if (len(data)//4) != ntimes:
            dtype = "d"

        data_az = struct.unpack(f"@{ntimes}{dtype}", data)
        out["az"] = np.array(data_az)
    
    with open(os.path.join(path, f"{chunk_idx}el.out"), 'rb') as fh:
        data = fh.read()
        data_el = struct.unpack(f"@{ntimes}{dtype}", data)
        out["el"] = np.array(data_el)
    
    with open(os.path.join(path, f"{chunk_idx}signal.out"), 'rb') as fh:
        data = fh.read()
        nsig = len(data)//_type2fmt[dtype]
        data_l = struct.unpack(f"@{nsig}{dtype}", data)
        nfreq = nsig // ntimes
        out["signal"] = np.array(data_l).reshape((nfreq, ntimes)).T

    return out
