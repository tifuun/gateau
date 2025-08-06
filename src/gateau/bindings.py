"""!
@file
Bindings for the ctypes interface for gateau. 
"""
from importlib import resources as impresources

from ctypes import (
    Structure,
    POINTER,
    c_float,
    c_int,
    c_char_p,
    c_ulonglong,
    CDLL
    )
import numpy as np
import os
import pathlib
from contextlib import ExitStack

import gateau.threadmgr as gmanager
import gateau.structs as gstructs
import gateau.bind_utils as gutils
from gateau import resources

def load_gateaulib() -> CDLL:
    """!
    Load the gateau shared library. Will detect the operating system and link the library accordingly.

    @returns The ctypes library containing the C/C++ functions.
    """

    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = CDLL(os.path.join(path_cur, "libgateau.dll"))
    except:
        try:
            lib = CDLL(os.path.join(path_cur, "libgateau.so"))
        except:
            lib = CDLL(os.path.join(path_cur, "libgateau.dylib"))

    lib.run_gateau.argtypes = [POINTER(gstructs.Instrument), 
                               POINTER(gstructs.Telescope),
                               POINTER(gstructs.Atmosphere), 
                               POINTER(gstructs.Source),
                               POINTER(gstructs.Cascade),
                               c_int, 
                               c_char_p,
                               c_ulonglong]
    
    lib.run_gateau.restype = None

    return lib

def run_gateau(instrument: dict[str, any], 
               telescope: dict[str, any], 
               atmosphere: dict[str, any], 
               source: dict[str, any], 
               cascade: dict[str, any], 
               nTimes: int, 
               outpath: str, 
               seed: int = 0,
               resourcepath: str | None = None, 
               ) -> None:
    """!
    Binding for running the gateau simulation on GPU.

    @param instrument Dictionary containing instrument parameters.
    @param telescope Dictionary containing telescope parameters.
    @param atmosphere Dictionary containing atmosphere parameters.
    @param source Dictionary containing astronomical source parameters.
    @param nTimes Number of time evaluations.
    @param outpath Path to directory where gateau output is stored.
    @param resourcepath Path to resources folder (None for autodetect)

    @returns 2D array containing timestreams of power in detector, for each channel frequency
    """

    lib = load_gateaulib()
    mgr = gmanager.Manager()

    _instrument = gstructs.Instrument()
    _telescope = gstructs.Telescope()
    _atmosphere = gstructs.Atmosphere()
    _source = gstructs.Source()
    _cascade = gstructs.Cascade()

    gutils.allfillInstrument(instrument, _instrument)
    gutils.allfillTelescope(telescope, _telescope)
    gutils.allfillAtmosphere(atmosphere, _atmosphere)
    gutils.allfillSource(source, _source)
    gutils.allfillCascade(cascade, _cascade)

    cnTimes = c_int(nTimes)
    coutpath = c_char_p(outpath.encode())
    cseed = c_ulonglong(seed)

    size_out = nTimes * instrument["nf_ch"]

    with ExitStack() as stack:
        if resourcepath is None:
            resourcepath = stack.enter_context(
                impresources.path(resources)
                )

        args = [
            _instrument,
            _telescope,
            _atmosphere,
            _source,
            _cascade,
            cnTimes,
            coutpath,
            cseed,
            resourcespath,
            ]

        # This blocks.
        mgr.new_thread(target=lib.run_gateau, args=args)

