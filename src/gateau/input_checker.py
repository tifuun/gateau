"""!
@file
Checker functions for input dictionaries.

Apart from checking, the functions also set default parameters that are not given by the user in the input dictionary.
"""

import numpy as np
import gateau.materials as gmaterials

TATM = 273
VWIND = 10
HCOLUMN = 1500
DXY = 0.2

def checkTelescopeDict(telescopeDict):
    checklist = ["eta_taper"]#, "az_scan", "el_scan"]

    errlist = []
        
    #for key in checklist:
    #    if telescopeDict.get(key) is None:
    #        errlist.append(key)

    return errlist

def checkInstrumentDict(instrumentDict):
    checklist = ["material", 
                 "f_sample",
                 "sec_harmonic",
                 "radius",
                 "eta_peak",
                 "use_filterbank"]

    errlist = []

    if instrumentDict.get("sec_harmonic") is None:
        instrumentDict["sec_harmonic"] = False

    if instrumentDict.get("pink_level") is None:
        instrumentDict["use_pink"] = 0

    else:
        instrumentDict["use_pink"] = 1

    if instrumentDict.get("pink_alpha") is None:
        instrumentDict["pink_alpha"] = 1

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
    
    if instrumentDict.get("use_filterbank") is None:
        instrumentDict["use_filterbank"] = True
    
    if instrumentDict.get("eta_peak") is None:
        instrumentDict["eta_peak"] = 1

    for key in checklist:
        if instrumentDict.get(key) is None:
            errlist.append(key)
    
    return errlist

def checkAtmosphereDict(atmosphereDict):
    checklist = ["path"]

    errlist = []

    if atmosphereDict.get("T_atm") is None:
        atmosphereDict["T_atm"] = TATM
    
    if atmosphereDict.get("v_wind") is None:
        atmosphereDict["v_wind"] = VWIND
    
    if atmosphereDict.get("h_column") is None:
        atmosphereDict["h_column"] = HCOLUMN

    if (PWV0 := atmosphereDict.get("PWV0")) is not None:
        if isinstance(PWV0, float) or isinstance(PWV0, int):
            atmosphereDict["PWV0"] = (PWV0, PWV0)
    else:
        atmosphereDict["PWV0"] = (1, 1)

    if atmosphereDict.get("dx") is None:
        atmosphereDict["dx"] = DXY
    
    if atmosphereDict.get("dy") is None:
        atmosphereDict["dy"] = DXY

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
    
