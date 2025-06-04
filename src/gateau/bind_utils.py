"""!
@file
File containing utility functions for the ctypes bindings.

Most of these functions are concerned with allocating memory.
"""

import numpy as np
import gateau.structs as gstructs
import ctypes
import os
import array as ar

def allfillCascade(CascadeDict, CascadeStruct, ct):
    """
    In plaats van dit, doe arrays maken met eta, T_parasitic, d, etc.
    Dan, gebruik order_refl en order_refr om voor elke cascade stap de juiste soort te gebruiken. Dan wordt de index voor de corresponderende type arrays geincrement.
    """
    eta_l = []
    T_parasitic_l = []
    d_l = []
    tandelta_l = []
    neff_l = []
    T_parasitic_reflect_l = []
    T_parasitic_refract_l = []
    use_AR_l = []
    
    order_refl = []
    order_refr = []

    for i, (item, key) in CascadeDict:
        if len(item.keys()) == 2: # Reflection dict
            eta_l.append(ct(item["eta"]))
            T_parasitic_l.append(ct(item["T_parasitic"]))
        
            order_refl.append(i)

        elif len(item.keys()) == 5: # Refraction dict
            d_l.append(ct(item["d"]))
            tandelta_l.append(ct(item["tandelta"]))
            neff_l.append(ct(item["neff"]))
            T_parasitic_reflect_l.append(ct(item["T_parasitic_reflect"]))
            T_parasitic_refract_l.append(ct(item["T_parasitic_refract"]))
            use_AR_l.append(ctypes.c_int(int(item["use_AR"])))

            order_refr.append(i)

    CascadeStruct.eta = (ct * len(order_refl))(*(eta_l))
    CascadeStruct.T_parasitic = (ct * len(order_refl))(*(T_parasitic_l))
    CascadeStruct.order_refl = (ctypes.c_int * len(order_refl))(*(order_refl))

    CascadeStruct.d = (ct * len(order_refr))(*(d_l))
    CascadeStruct.tandelta = (ct * len(order_refr))(*(tandelta_l))
    CascadeStruct.T_parasitic_refl = (ct * len(order_refr))(*(T_parasitic_reflect_l))
    CascadeStruct.T_parasitic_refr = (ct * len(order_refr))(*(T_parasitic_refract_l))
    CascadeStruct.use_AR = (ctypes.c_int * len(order_refr))(*(use_AR_l))
    CascadeStruct.order_refr = (ctypes.c_int * len(order_refr))(*(order_refr))

def allfillInstrument(InstDict, InstStruct, ct_t=ctypes.c_double):
    """!
    Allocate and fill an instrument struct for ctypes.
    
    @param InstDict Dictionary containing instrument parameters.
    @param InstStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """

    arr_t = 'd' if ct_t == ctypes.c_double else 'f'
    arr_filterbank = ar.array(arr_t, InstDict["filterbank"].ravel())

    df_src = (InstDict["f1_src"] - InstDict["f0_src"]) / InstDict["nf_src"]

    InstStruct.nf_ch = ctypes.c_int(InstDict["nf_ch"])

    InstStruct.f_spec = arr2ArrSpec(InstDict["f_src"], ct_t)

    InstStruct.f_sample = ct_t(InstDict["f_sample"])
    InstStruct.filterbank = (ct_t * InstDict["filterbank"].size).from_buffer(arr_filterbank)
    InstStruct.delta = ct_t(InstDict["delta"])
    InstStruct.eta_pb = ct_t(InstDict["eta_pb"])

def allfillTelescope(TelDict, TelStruct, ct_t=ctypes.c_double):
    """!
    Allocate and fill a telescope struct for ctypes.
    
    @param TelDict Dictionary containing telescope parameters.
    @param TelStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """
    
    TelStruct.Dtel = ct_t(TelDict["Dtel"])
    TelStruct.chop_mode = ctypes.c_int(TelDict["chop_mode"])
    TelStruct.dAz_chop = ct_t(TelDict["dAz_chop"])
    TelStruct.freq_chop = ct_t(TelDict["freq_chop"])
    TelStruct.freq_nod = ct_t(TelDict["freq_nod"])
    TelStruct.eta_ap_ON = (ct_t * TelDict["eta_ap_ON"].size)(*(TelDict["eta_ap_ON"].ravel().tolist()))
    TelStruct.eta_ap_OFF = (ct_t * TelDict["eta_ap_OFF"].size)(*(TelDict["eta_ap_OFF"].ravel().tolist()))

    TelStruct.scantype = ctypes.c_int(TelDict["scantype"])
    TelStruct.El0 = ct_t(TelDict["El0"])
    TelStruct.Ax = ct_t(TelDict["Ax"])
    TelStruct.Axmin = ct_t(TelDict["Axmin"])
    TelStruct.Ay = ct_t(TelDict["Ay"])
    TelStruct.Aymin = ct_t(TelDict["Aymin"])
    TelStruct.wx = ct_t(TelDict["wx"])
    TelStruct.wxmin = ct_t(TelDict["wxmin"])
    TelStruct.wy = ct_t(TelDict["wy"])
    TelStruct.wymin = ct_t(TelDict["wymin"])
    TelStruct.phix = ct_t(TelDict["phix"])
    TelStruct.phiy = ct_t(TelDict["phiy"])


def allfillAtmosphere(AtmDict, AtmStruct, ct_t=ctypes.c_double, coalesce=False):
    """!
    Allocate and fill an atmosphere struct for ctypes.
    
    @param AtmDict Dictionary containing atmosphere parameters.
    @param AtmStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """

    AtmStruct.Tatm = ct_t(AtmDict["Tatm"])
    AtmStruct.v_wind = ct_t(AtmDict["v_wind"])
    AtmStruct.h_column = ct_t(AtmDict["h_column"])
    AtmStruct.dx = ct_t(AtmDict["dx"])
    AtmStruct.dy = ct_t(AtmDict["dy"])
    AtmStruct.path = ctypes.c_char_p(os.path.join(AtmDict["path"], "prepd").encode())

def allfillSource(SourceDict, SourceStruct, ct_t=ctypes.c_double):
    """!
    Allocate and fill a source object struct for ctypes.
    
    @param SourceDict Dictionary containing source angular extents and intensity maps.
    @param SourceStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """
       
    I_nu = SourceDict["I_nu"]

    nI_nu = I_nu.ravel().size

    SourceStruct.Az_spec = arr2ArrSpec(SourceDict["Az_src"], ct_t)
    SourceStruct.El_spec = arr2ArrSpec(SourceDict["El_src"], ct_t)
    
    SourceStruct.I_nu = (ct_t * nI_nu)(*(I_nu.ravel().tolist())) 
    SourceStruct.nI_nu = ctypes.c_int(nI_nu)

def allfillArrSpec(arr, ArrSpecStruct, ct_t=ctypes.c_double):
    ArrSpecStruct.start = ct_t(arr[0])    
    ArrSpecStruct.step = ct_t(arr[1] - arr[0])    
    ArrSpecStruct.num = ctypes.c_int(arr.size)    

def arr2ArrSpec(arr, ct_t=ctypes.c_double):
    arrspec = gstructs.ArrSpec(ct_t)
    allfillArrSpec(arr, arrspec, ct_t)

    return arrspec





