"""!
@file
Checker functions for input dictionaries.

Apart from checking, the functions also set default parameters that are not given by the user in the input dictionary.
"""

import numpy as np
import tiempo2.Materials as TMaterials

def checkTelescopeDict(telescopeDict):
    checklist = ["Dtel", "Ttel", "Tgnd", "chop_mode", "eta_ap_ON", "eta_mir",
                     "eta_fwd", "freq_chop", "dAz_chop", "El0"]

    checklist_daisy = ["Ax", "Axmin", "Ay", "Aymin", "wx", "wxmin", "wy", "wymin"]

    errlist = []


    if telescopeDict.get("chop_mode") is None:
        telescopeDict["chop_mode"] = 0
        telescopeDict["freq_chop"] = -1
        telescopeDict["dAz_chop"] = -1
        telescopeDict["freq_nod"] = -1
    elif telescopeDict.get("chop_mode") == "direct":
        telescopeDict["chop_mode"] = 1
    elif telescopeDict.get("chop_mode") == "abba":
        telescopeDict["chop_mode"] = 2

    if telescopeDict.get("El0") is None:
        telescopeDict["El0"] = 90.0

    if telescopeDict.get("scantype") is None:
        telescopeDict["scantype"] = "point"
        for item in checklist_daisy:
            telescopeDict[item] = 0
            telescopeDict["phix"] = 0
            telescopeDict["phiy"] = 0
        telescopeDict["scantype"] = 0

    elif telescopeDict.get("scantype") == "daisy":
        if telescopeDict.get("phix") is None:
            telescopeDict["phix"] = 0
        if telescopeDict.get("phiy") is None:
            telescopeDict["phiy"] = 0
        
        for key in checklist_daisy:
            if telescopeDict.get(key) is None:
                errlist.append(key)

        telescopeDict["scantype"] = 1

    elif telescopeDict.get("scantype") == "point":
        for key in checklist_daisy:
            telescopeDict[key] = 0
        telescopeDict["scantype"] = 0

    for key in checklist:
        if telescopeDict.get(key) is None:
            errlist.append(key)


    return errlist

def checkInstrumentDict(instrumentDict):
    checklist = ["material", "f0_ch", "eta_inst", "f_sample", "eta_filt", "box_eq", "order"]

    errlist = []

    fail = False

    if isinstance(instrumentDict.get("f0_ch"), np.ndarray):
        instrumentDict["nf_ch"] = instrumentDict.get("f0_ch").size

    if instrumentDict.get("box_eq") is None:
        instrumentDict["box_eq"] = True

    if instrumentDict.get("order") is None:
        instrumentDict["order"] = 1

    if isinstance(instrumentDict.get("eta_filt"), float):
        instrumentDict["eta_filt"] *= np.ones(instrumentDict.get("nf_ch"))

    if instrumentDict.get("material") is None:
        instrumentDict["material"] = "Al_NbTiN"
    
    if instrumentDict.get("material") == "Al_NbTiN":
        instrumentDict["delta"] = TMaterials.Al_NbTiN["delta"]
        instrumentDict["eta_pb"] = TMaterials.Al_NbTiN["eta_pb"]

    else:
        errlist.append("material")
    
    for key in checklist:
        if instrumentDict.get(key) is None:
            errlist.append(key)
            fail = True
    

    return errlist

def checkAtmosphereDict(atmosphereDict):
    checklist = ["Tatm", "filename", "path", "dx", "dy", "h_column", "v_wind", "PWV0"]

    errlist = []

    for key in checklist:
        if atmosphereDict.get(key) is None:
            errlist.append(key)
    
    return errlist
