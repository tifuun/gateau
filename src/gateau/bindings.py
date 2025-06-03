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

def run_gateau(instrument, telescope, atmosphere, source, cascade, nTimes, outpath):
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

    lib, ct = load_gateaulib()
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

