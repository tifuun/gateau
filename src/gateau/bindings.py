"""!
@file
Bindings for the ctypes interface for gateau. 
"""

from ctypes import Structure, POINTER, c_float, c_int, c_char_p
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

    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libcugateau.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libcugateau.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libcugateau.dylib"))

    lib.run_gateau.argtypes = [POINTER(gstructs.Instrument), 
                               POINTER(gstructs.Telescope),
                               POINTER(gstructs.Atmosphere), 
                               POINTER(gstructs.Source),
                               POINTER(gstructs.Cascade),
                               c_int, 
                               c_char_p]
    
    lib.run_gateau.restype = None

    return lib

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

    lib = load_gateaulib()
    mgr = gmanager.Manager()

    _instrument = gstructs.Instrument()
    _telescope = gstructs.Telescope()
    _atmosphere = gstructs.Atmosphere()
    _source = gstructs.Source()
    _cascade = gstructs.Cascade()

    gutils.allfillInstrument(instrument, _instrument)
    gutils.allfillTelescope(telescope, _telescope)
    start = time.time()
    gutils.allfillAtmosphere(atmosphere, _atmosphere)
    end = time.time()
    gutils.allfillSource(source, _source)
    gutils.allfillCascade(cascade, _cascade)

    cnTimes = c_int(nTimes)
    coutpath = c_char_p(outpath.encode())

    size_out = nTimes * instrument["nf_ch"]

    timed = end-start

    args = [_instrument, _telescope, _atmosphere, _source, _cascade, cnTimes, coutpath]

    mgr.new_thread(target=lib.run_gateau, args=args)

