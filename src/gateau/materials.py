"""!
@file 
File containing MKID material (combinations).
Users specify the type of MKID from the instrument dictionary.
These material dicts then give the gap energy and pair breaking efficiency.
Also, TLS noise characteristics are set from this file.
"""

Al_NbTiN = {
        "delta"         : 188 * 1.60218e-19 * 1e-6,
        "eta_pb"        : 0.4,
        "Tc"            : 1.2,
        "N0"            : 1.72 * 1e4 * 1e6 * (1e6)**3 * 1.60218e19,
        "cutoff"        : 90e9
        }

