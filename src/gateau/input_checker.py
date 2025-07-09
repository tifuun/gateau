"""!
@file
Checker functions for input dictionaries.

Apart from checking, the functions also set default parameters that are not given by the user in the input dictionary.
"""

import numpy as np
import gateau.materials as gmaterials

def checkTelescopeDict(telescopeDict):
    checklist = ["eta_ap"]#, "az_scan", "el_scan"]

    errlist = []
        
    for key in checklist:
        if telescopeDict.get(key) is None:
            errlist.append(key)

    return errlist

def checkInstrumentDict(instrumentDict):
    checklist = ["material", "f0_ch", "f_sample", "box_eq", "order", "radius"]

    errlist = []

    fail = False

    # f0_ch is array of center frequencies
    if isinstance(instrumentDict.get("f0_ch"), np.ndarray):
        instrumentDict["nf_ch"] = instrumentDict.get("f0_ch").size

    if instrumentDict.get("box_eq") is None:
        instrumentDict["box_eq"] = True

    if instrumentDict.get("order") is None:
        instrumentDict["order"] = 1

    if instrumentDict.get("material") is None:
        instrumentDict["material"] = "Al_NbTiN"
    
    if instrumentDict.get("material") == "Al_NbTiN":
        instrumentDict["delta"] = gmaterials.Al_NbTiN["delta"]
        instrumentDict["eta_pb"] = gmaterials.Al_NbTiN["eta_pb"]
        instrumentDict["cutoff"] = gmaterials.Al_NbTiN["cutoff"]


    else:
        errlist.append("material")

    if instrumentDict.get("radius") is None:
        instrumentDict["radius"] = 0

    for key in checklist:
        if instrumentDict.get(key) is None:
            errlist.append(key)
    
    return errlist

def checkAtmosphereDict(atmosphereDict):
    checklist = ["T_atm", "filename", "path", "dx", "dy", "h_column", "v_wind"]

    errlist = []

    for key in checklist:
        if atmosphereDict.get(key) is None:
            errlist.append(key)
    
    return errlist

def checkSourceDict(sourceDict):
    checklist = ["I_nu", "az_src", "el_src", "f_src"]
    
    errlist = []

    for key in checklist:
        if sourceDict.get(key) is None:
            errlist.append(key)
    
    return errlist
    
