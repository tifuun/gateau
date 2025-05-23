import os
import numpy as np

AtmDict = {
        "Tatm"          : 273,
        "filename"      : "sample00.dat",
        "path"          : os.path.join(os.path.dirname(os.path.abspath(__file__)), "aris"),
        "dx"            : 0.2,
        "dy"            : 0.2,
        "h_column"      : 1000,
        "v_wind"        : 10,
        "PWV0"          : 1
        }

TelDict = {
        "Dtel"          : 10,
        "Ttel"          : 300,
        "Tgnd"          : 280,
        "eta_ap_ON"     : 0.66,
        "eta_ap_OFF"    : 0.66,
        "eta_mir"       : 0.89,
        "eta_fwd"       : 0.9,
        "freq_nod"      : 4/60,
        "freq_chop"     : 10,
        "s_rms"         : 42,
        "chop_mode"     : "abba",
        "dAz_chop"      : 234
        }

InstDict = {
        "f0_ch"         : 200,
        "R"             : 500,
        "nf_ch"         : 350,
        "eta_inst"      : 0.35,
        "eta_misc"      : 0.65, #eta_co
        "eta_filt"      : 0.3,
        "box_eq"        : False,
        "f_sample"      : 158,
        "f0_src"        : 190,
        "f1_src"        : 440,
        "nf_src"        : 2000,
        "material"      : "Al_NbTiN"
}

SourceDict = {
        "type"          : "SZ",
        "Az"            : np.array([-1, 1])* 240,
        "El"            : np.array([-1, 1])* 240,
        "nAz"           : 31,
        "nEl"           : 31,
        "Te"            : 15.3,
        "ne0"           : 1e-2,
        "beta"          : 0.73,
        "v_pec"         : 2500,
        "rc"            : 116,
        "thetac"        : 40.71,
        "Da"            : 760,
}


