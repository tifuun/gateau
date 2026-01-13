"""!
@file
Bindings for the ctypes interface for gateau. 
"""
from importlib import resources as impresources

from ctypes import (
    POINTER,
    c_int,
    c_char_p,
    c_ulonglong,
    CDLL
    )
from typing import Union
from contextlib import ExitStack
import platform

import gateau.threadmgr as gmanager
import gateau.structs as gstructs
import gateau.bind_utils as gutils

import gateau

def load_gateaulib() -> CDLL:
    """!
    Load the gateau shared library. Will detect the operating system and link the library accordingly.

    @returns The ctypes library containing the C/C++ functions.
    """

    if platform.system() == "Windows":
        lib_filename = "gateau.dll"
    elif platform.system() == "Darwin":
        raise NotImplementedError(
            "Mac OS is not supported by Glateau and never will be. Sorry."
            )
    else:
        lib_filename = "libgateau.so"

    try:

        with impresources.path(gateau, lib_filename) as sopath:
            lib = CDLL(sopath)


    except OSError as err:
        raise OSError(
            f"Could not load `{lib_filename}`!! Did it fail to compile? "
            "Is it compiled for the wrong architecture? Is the file "
            "missing? It should be under the root of the `gateau` "
            f"package, `src/gateau/{lib_filename}` if you ran pip with "
            "`-e`. "
            ) from err

    lib.run_gateau.argtypes = [POINTER(gstructs.Instrument), 
                               POINTER(gstructs.Telescope),
                               POINTER(gstructs.Atmosphere), 
                               POINTER(gstructs.Source),
                               POINTER(gstructs.Cascade),
                               c_int, 
                               c_char_p,
                               c_char_p,
                               c_ulonglong,
                               c_char_p,
                               ]
    
    lib.run_gateau.restype = None

    return lib

# TODO fix Any!!!

def run_gateau(instrument: dict[str, any], 
               telescope: dict[str, any], 
               atmosphere: dict[str, any], 
               source: dict[str, any], 
               cascade: dict[str, any], 
               n_times: int, 
               outpath: str,
               outscale: str,
               seed: int,
               resourcepath: Union[str, None] = None, 
               ) -> None:
    """!
    Binding for running the gateau simulation.
    During the simulation, gateau will serialise the output into the specified folder.

    @param instrument Dictionary containing instrument parameters.
    @param telescope Dictionary containing telescope parameters.
    @param atmosphere Dictionary containing atmosphere parameters.
    @param source Dictionary containing astronomical source parameters.
    @param n_times Number of time evaluations.
    @param outpath Path to directory where gateau output is stored.
    @param outscale Scale of stored output (brightness temperature or power).
    @param seed Seed to use for noise calculations.
    @param resourcepath Path to resources folder (None for autodetect)
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

    cn_times = c_int(n_times)
    coutpath = c_char_p(str(outpath).encode())
    coutscale = c_char_p(outscale.encode())
    # FIXME bare encode
    cseed = c_ulonglong(seed)

    with ExitStack() as stack:
        if resourcepath is None:
            atmpath = stack.enter_context(
                impresources.path(gateau.resources, "eta_atm")
                )

        catmpath = c_char_p(str(atmpath).encode('utf-8'))

        args = [
            _instrument,
            _telescope,
            _atmosphere,
            _source,
            _cascade,
            cn_times,
            coutpath,
            coutscale,
            cseed,
            catmpath,
            ]

        # This blocks.
        # Or does it...??
        mgr.new_thread(target=lib.run_gateau, args=args)

