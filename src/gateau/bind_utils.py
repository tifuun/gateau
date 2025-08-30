"""!
@file
File containing utility functions for the ctypes bindings.

Most of these functions are concerned with allocating memory.
"""
from ctypes import Structure, c_float, c_int, c_char_p

import numpy as np
import gateau.structs as gstructs
import os
import array as ar

def allfillCascade(CascadeDict: dict[str, any], 
                   CascadeStruct: Structure) -> None:
    """!
    """
    arr_eta = ar.array('f', CascadeDict["eta_stage"].ravel())
    arr_psd = ar.array('f', CascadeDict["psd_stage"].ravel())
    
    CascadeStruct.eta_stage = (c_float * CascadeDict["eta_stage"].size).from_buffer(arr_eta)
    CascadeStruct.psd_stage = (c_float * CascadeDict["psd_stage"].size).from_buffer(arr_psd)
    CascadeStruct.num_stage = c_int(CascadeDict["num_stage"])

def allfillInstrument(InstDict: dict[str, any], 
                      InstStruct: Structure) -> None:
    """!
    Allocate and fill an instrument struct for ctypes.
    
    @param InstDict Dictionary containing instrument parameters.
    @param InstStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """

    arr_filterbank = ar.array('f', InstDict["filterbank"].ravel())
    arr_az_fpa = ar.array('f', InstDict["pointings"][0].ravel())
    arr_el_fpa = ar.array('f', InstDict["pointings"][1].ravel())

    InstStruct.nf_ch = c_int(InstDict["nf_ch"])
    InstStruct.f_sample = c_float(InstDict["f_sample"])
    InstStruct.filterbank = (c_float * InstDict["filterbank"].size).from_buffer(arr_filterbank)
    InstStruct.delta = c_float(InstDict["delta"])
    InstStruct.eta_pb = c_float(InstDict["eta_pb"])
    InstStruct.az_fpa = (c_float * InstDict["pointings"][0].size).from_buffer(arr_az_fpa)
    InstStruct.el_fpa = (c_float * InstDict["pointings"][1].size).from_buffer(arr_el_fpa)
    InstStruct.num_spax = c_int(InstDict["pointings"][0].size)

def allfillTelescope(TelDict: dict[str, any], 
                     TelStruct: Structure) -> None:
    """!
    Allocate and fill a telescope struct for ctypes.
    
    @param TelDict Dictionary containing telescope parameters.
    @param TelStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """
    arr_eta_ap = ar.array('f', TelDict["eta_ap"].ravel())
    arr_az_scan = ar.array('f', TelDict["az_scan"].ravel())
    arr_el_scan = ar.array('f', TelDict["el_scan"].ravel())
    
    TelStruct.eta_ap = (c_float * TelDict["eta_ap"].size).from_buffer(arr_eta_ap)
    TelStruct.az_scan = (c_float * TelDict["az_scan"].size).from_buffer(arr_az_scan)
    TelStruct.el_scan = (c_float * TelDict["el_scan"].size).from_buffer(arr_el_scan)

def allfillAtmosphere(AtmDict: dict[str, any], 
                      AtmStruct: Structure) -> None:
    """!
    Allocate and fill an atmosphere struct for ctypes.
    
    @param AtmDict Dictionary containing atmosphere parameters.
    @param AtmStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """

    AtmStruct.Tatm = c_float(AtmDict["T_atm"])
    AtmStruct.v_wind = c_float(AtmDict["v_wind"])
    AtmStruct.h_column = c_float(AtmDict["h_column"])
    AtmStruct.dx = c_float(AtmDict["dx"])
    AtmStruct.dy = c_float(AtmDict["dy"])
    AtmStruct.path = c_char_p(os.path.join(AtmDict["path"], "prepd").encode())

def allfillSource(SourceDict: dict[str, any], 
                  SourceStruct: Structure) -> None:
    """!
    Allocate and fill a source object struct for ctypes.
    
    @param SourceDict Dictionary containing source angular extents and intensity maps.
    @param SourceStruct Struct to be filled and passed to ctypes.
    @param ct_t Type of data. Use ctypes.c_double for CPU, ctypes.c_float for GPU.
    """
    
    arr_I_nu = ar.array('f', SourceDict["I_nu"].ravel())
    nI_nu = SourceDict["I_nu"].size

    SourceStruct.az_src_spec = arr2ArrSpec(SourceDict["az_src"])
    SourceStruct.el_src_spec = arr2ArrSpec(SourceDict["el_src"])
    SourceStruct.f_spec = arr2ArrSpec(SourceDict["f_src"])
    SourceStruct.I_nu = (c_float * SourceDict["I_nu"].size).from_buffer(arr_I_nu) 
    SourceStruct.nI_nu = c_int(SourceDict["I_nu"].size)

def allfillArrSpec(arr: np.ndarray, 
                   ArrSpecStruct: Structure) -> None:
    ArrSpecStruct.start = c_float(arr[0])    
    ArrSpecStruct.step = c_float(arr[1] - arr[0])    
    ArrSpecStruct.num = c_int(arr.size)    

def arr2ArrSpec(arr: np.ndarray) -> Structure:
    arrspec = gstructs.ArrSpec()
    allfillArrSpec(arr, arrspec)

    return arrspec
