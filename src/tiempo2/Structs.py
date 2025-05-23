"""!
@file
Structs that are passed to the C++ backend.
""" 

import ctypes
import numpy as np

class ArrSpec(ctypes.Structure):
    """!
    Struct used for passing array specifications a0, da, na.
    Used when it is not necessary to pass full dictionaries.
    """

    def __init__(self, ct):
        self._fields_ = [("start", ct),
                    ("step", ct),
                    ("num", ctypes.c_int)]

class CalOutput(ctypes.Structure):
    """!
    Struct used as output for a power to temperature conversion database.
    """

    def __init__(self, ct):
        self._fields_ = [("power", ctypes.POINTER(ct)),
                ("temperature", ctypes.POINTER(ct))]

class Cascade(ctypes.Structure):
    """!
    Struct for storing all cascades
    """
    def __init__(self, ct):
        self._fields_ = [("eta", ctypes.POINTER(ct)),
                         ("T_parasitic", ctypes.POINTER(ct)),
                         ("d", ctypes.POINTER(ct)), 
                         ("tandelta", ctypes.POINTER(ct)),
                         ("neff", ctypes.POINTER(ct)),
                         ("T_parasitic_refl", ctypes.POINTER(ct)),
                         ("T_parasitic_refr", ctypes.POINTER(ct)),
                         ("use_AR", ctypes.POINTER(ctypes.c_int)),
                         ("order_refl", ctypes.POINTER(ctypes.c_int)),
                         ("order_refr", ctypes.POINTER(ctypes.c_int))]

class Instrument(ctypes.Structure):
    """!
    Struct representing the simulated instrument.
    """

    def __init__(self, ct):
        self._fields_ = [("nf_ch", ctypes.c_int),
                ("f_spec", ArrSpec(ct)),
                ("f_sample", ct),
                ("filterbank", ctypes.POINTER(ct)),
                ("delta", ct),
                ("eta_pb", ct)]

class Telescope(ctypes.Structure):
    """!
    Struct representing the simulated telescope.
    """

    def __init__(self, ct):
        self._fields_ = [
                ("Dtel", ct),
                ("chop_mode", ctypes.c_int),
                ("dAz_chop", ct),
                ("freq_chop", ct),
                ("freq_nod", ct),
                ("eta_ap_ON", ctypes.POINTER(ct)),
                ("eta_ap_OFF", ctypes.POINTER(ct)),
                ("scantype", ctypes.c_int),
                ("El0", ct),
                ("Ax", ct),
                ("Axmin", ct),
                ("Ay", ct),
                ("Aymin", ct),
                ("wx", ct),
                ("wxmin", ct),
                ("wy", ct),
                ("wymin", ct),
                ("phix", ct),
                ("phiy", ct)]


class Atmosphere(ctypes.Structure):
    """!
    Struct representing the simulated atmosphere.
    """

    def __init__(self, ct):
        self._fields_ = [("Tatm", ct),
                ("v_wind", ct),
                ("h_column", ct),
                ("dx", ct),
                ("dy", ct),
                ("path", ctypes.c_char_p)]

class Source(ctypes.Structure):
    """!
    Struct representing simulated astronomical source.
    """

    def __init__(self, ct):
        self._fields_ = [("Az_spec", ArrSpec(ct)),
                ("El_spec", ArrSpec(ct)),
                ("I_nu", ctypes.POINTER(ct)),
                ("nI_nu", ctypes.c_int)]

