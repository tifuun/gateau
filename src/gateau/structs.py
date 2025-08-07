"""!
@file
Structs that are passed to the C++ backend.
""" 

from ctypes import Structure, POINTER, c_float, c_int, c_char_p

class ArrSpec(Structure):
    """!
    Struct used for passing regular array specifications: start, step, num.
    Used when it is not necessary to pass full dictionaries.
    """

    _fields_ = [("start", c_float),
                ("step", c_float),
                ("num", c_int)]

class Cascade(Structure):
    """!
    Struct for storing all (grouped) stages in a cascade.
    """

    _fields_ = [("eta_stage", POINTER(c_float)),
                 ("psd_stage", POINTER(c_float)),
                 ("num_stage", c_int)]

class Instrument(Structure):
    """!
    Struct representing the simulated instrument.
    """

    _fields_ = [("nf_ch", c_int),
                ("f_sample", c_float),
                ("filterbank", POINTER(c_float)),
                ("delta", c_float),
                ("eta_pb", c_float),
                ("az_fpa", POINTER(c_float)),
                ("el_fpa", POINTER(c_float)),
                ("num_spax", c_int)]

class Telescope(Structure):
    """!
    Struct representing the simulated telescope.
    """

    _fields_ = [
            ("eta_ap", POINTER(c_float)),
            ("az_scan", POINTER(c_float)),
            ("el_scan", POINTER(c_float))]

class Atmosphere(Structure):
    """!
    Struct representing the simulated atmosphere.
    """

    _fields_ = [("Tatm", c_float),
            ("v_wind", c_float),
            ("h_column", c_float),
            ("dx", c_float),
            ("dy", c_float),
            ("path", c_char_p)]

class Source(Structure):
    """!
    Struct representing simulated astronomical source.
    """

    _fields_ = [("az_src_spec", ArrSpec),
            ("el_src_spec", ArrSpec),
            ("f_spec", ArrSpec),
            ("I_nu", POINTER(c_float)),
            ("nI_nu", c_int)]

