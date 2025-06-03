"""!
@file
Bindings for the ctypes interface for gateau. 
"""

import ctypes
import numpy as np
import os
import pathlib

import gateau.threadmgr as gmanager
import gateau.structs as gstructs
import gateau.bind_utils as gutils

def load_gateaulib():
    """!
    Load the gateau shared library. Will detect the operating system and link the library accordingly.

    @returns lib The ctypes library containing the C/C++ functions.
    """

    ct = ctypes.c_double

    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libgateau.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libgateau.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libgateau.dylib"))

    lib.calcW2K.argtypes = [ctypes.POINTER(gstructs.Instrument(ct)), 
                            ctypes.POINTER(gstructs.Telescope(ct)),
                            ctypes.POINTER(gstructs.Atmosphere(ct)), 
                            ctypes.POINTER(gstructs.CalOutput(ct)),
                            ctypes.c_int, ctypes.c_int]
    
    lib.getSourceSignal.argtypes = [ctypes.POINTER(gstructs.Instrument(ct)), 
                                    ctypes.POINTER(gstructs.Telescope(ct)),
                                    ctypes.POINTER(ct), ctypes.POINTER(ct),
                                    ct, ctypes.c_bool]

    lib.getEtaAtm.argtypes = [gstructs.ArrSpec(ct), ctypes.POINTER(ct), ct]
    
    lib.getNEP.argtypes = [ctypes.POINTER(gstructs.Instrument(ct)), 
                           ctypes.POINTER(gstructs.Telescope(ct)),
                           ctypes.POINTER(ct), ct, ct]
    
    lib.calcW2K.restype = None
    lib.getSourceSignal.restype = None
    lib.getEtaAtm.restype = None
    lib.getNEP.restype = None

    return lib, ct

def load_gateaulib_CUDA():
    """!
    Load the gateau shared library. Will detect the operating system and link the library accordingly.

    @returns The ctypes library containing the C/C++ functions.
    """
    
    ct = ctypes.c_float

    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libcugateau.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libcugateau.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libcugateau.dylib"))

    lib.run_gateau.argtypes = [ctypes.POINTER(gstructs.Instrument(ct)), 
                                    ctypes.POINTER(gstructs.Telescope(ct)),
                                    ctypes.POINTER(gstructs.Atmosphere(ct)), 
                                    ctypes.POINTER(gstructs.Source(ct)),
                                    ctypes.POINTER(gstructs.Cascade(ct)),
                                    ctypes.c_int, ctypes.c_char_p]
    
    lib.run_gateau.restype = None

    return lib, ct

def calcW2K(instrument, telescope, atmosphere, nPWV, nThreads):
    """!
    Binding for calculating a power-temperature conversion table.

    @param instrument Dictionary containing instrument parameters.
    @param telescope Dictionary containing telescope parameters.
    @param atmosphere Dictionary containing atmosphere parameters.
    @param w2k Watt to Kelvin specification parameters.

    @returns A CalOutput dictionary, containing power-temperature relations.
    """

    lib, ct = load_gateaulib()
    mgr = gmanager.Manager()

    _instrument = gstructs.Instrument(ct)
    _telescope = gstructs.Telescope(ct)
    _atmosphere = gstructs.Atmosphere(ct)

    _caloutput = gstructs.CalOutput(ct)

    gutils.allfillInstrument(instrument, _instrument)
    gutils.allfillTelescope(telescope, _telescope)
    gutils.allfillAtmosphere(atmosphere, _atmosphere)

    gutils.allocateCalOutput(_caloutput, nPWV, instrument["nf_ch"])

    cnPWV = ctypes.c_int(nPWV)
    cnThreads = ctypes.c_int(nThreads)

    args = [_instrument, _telescope, _atmosphere, _caloutput, cnPWV, cnThreads]

    mgr.new_thread(target=lib.calcW2K, args=args)

    res = gutils.CalOutputStructToDict(_caloutput, nPWV, instrument["nf_ch"], np_t=np.float64)

    return res

def getSourceSignal(instrument, telescope, atmosphere, I_nu, PWV, ON):
    """!
    Binding for calculating the source signal, through the optical path, but without noise.

    @param instrument Dictionary containing instrument parameters.
    @param telescope Dictionary containing telescope parameters.
    @param atmosphere Dictionary containing atmosphere parameters.
    @param source Dictionary containing astronomical source parameters.
    @param Az_point Azimuth point on-sky at which to calculate source intensity.
    @param El_point Elevation point on-sky at which to calculate source intensity.
    @param PWV PWV value of atmosphere, in mm. If empty, no atmosphere used.
    @param ON Use ON-path. If False, uses OFF path.

    @returns 1D array containing power for each detector.
    """

    lib, ct = load_gateaulib()
    mgr = gmanager.Manager()

    _instrument = gstructs.Instrument(ct)
    _telescope = gstructs.Telescope(ct)

    coutput = (ct * instrument["nf_ch"]).from_buffer(np.zeros(instrument["nf_ch"]))

    cI_nu = (ct * I_nu.size)(*(I_nu.ravel().tolist()))

    cPWV = ct(PWV)
    cON = ctypes.c_bool(ON)

    gutils.allfillInstrument(instrument, _instrument)
    gutils.allfillTelescope(telescope, _telescope)
    
    args = [_instrument, _telescope, coutput, cI_nu, cPWV, cON]

    mgr.new_thread(target=lib.getSourceSignal, args=args)

    res = np.ctypeslib.as_array(coutput, shape=instrument["nf_ch"]).astype(np.float64)
    
    return res

def getEtaAtm(instrument, PWV):
    """!
    Binding for running the gateau simulation.

    @param source Dictionary containing astronomical source parameters.
    @param atmosphere Dictionary containing atmosphere parameters.
    @param PWV PWV value of atmosphere, in mm.
    
    @returns 1D array containing atmospheric transmission for each detector.
    """


    lib, ct = load_gateaulib()
    mgr = gmanager.Manager()

    coutput = (ct * instrument["nf_src"])(*(np.zeros(instrument["nf_src"]).tolist()))

    cPWV = ct(PWV)

    f_src_spec = gutils.arr2ArrSpec(instrument["f_src"])

    args = [f_src_spec, coutput, cPWV]

    mgr.new_thread(target=lib.getEtaAtm, args=args)

    res = np.ctypeslib.as_array(coutput, shape=instrument["nf_src"]).astype(np.float64)
    
    return res

def getNEP(instrument, telescope, atmosphere, PWV):
    """!
    Binding for running the gateau simulation.

    @param instrument Dictionary containing instrument parameters.
    @param telescope Dictionary containing telescope parameters.
    @param atmosphere Dictionary containing atmosphere parameters.
    @param PWV PWV value of atmosphere, in mm.
    
    @returns 1D array containing NEP for each detector.
    """

    lib, ct = load_gateaulib()
    mgr = gmanager.Manager()

    _instrument = gstructs.Instrument(ct)
    _telescope = gstructs.Telescope(ct)
    
    coutput = (ct * instrument["nf_ch"]).from_buffer(np.zeros(instrument["nf_ch"]))

    cPWV = ct(PWV)
    cTatm = ct(atmosphere["Tatm"])

    gutils.allfillInstrument(instrument, _instrument)
    gutils.allfillTelescope(telescope, _telescope)

    args = [_instrument, _telescope, coutput, cPWV, cTatm]

    mgr.new_thread(target=lib.getNEP, args=args)

    res = np.ctypeslib.as_array(coutput, shape=instrument["nf_ch"]).astype(np.float64)
    
    return res

def rungateau_CUDA(instrument, telescope, atmosphere, source, cascade, nTimes, outpath):
    """!
    Binding for running the gateau simulation on GPU.

    @param instrument Dictionary containing instrument parameters.
    @param telescope Dictionary containing telescope parameters.
    @param atmosphere Dictionary containing atmosphere parameters.
    @param source Dictionary containing astronomical source parameters.
    @param nTimes Number of time evaluations.
    @param outpath Path to directory where gateau output is stored.

    @returns 2D array containing timestreams of power in detector, for each channel frequency
    """
    import time


    lib, ct = load_gateaulib_CUDA()
    mgr = gmanager.Manager()

    _instrument = gstructs.Instrument(ct)
    _telescope = gstructs.Telescope(ct)
    _atmosphere = gstructs.Atmosphere(ct)
    _source = gstructs.Source(ct)
    _cascade = gstructs.Cascade(ct)

    gutils.allfillInstrument(instrument, _instrument, ct)
    gutils.allfillTelescope(telescope, _telescope, ct)
    start = time.time()
    gutils.allfillAtmosphere(atmosphere, _atmosphere, ct, coalesce=True)
    end = time.time()
    gutils.allfillSource(source, _source, ct)
    gutils.allfillCascade(cascade, _cascade, ct)

    cnTimes = ctypes.c_int(nTimes)
    coutpath = ctypes.c_char_p(outpath.encode())

    size_out = nTimes * instrument["nf_ch"]

    timed = end-start

    args = [_instrument, _telescope, _atmosphere, _source, _cascade, cnTimes, coutpath]

    mgr.new_thread(target=lib.run_gateau, args=args)

