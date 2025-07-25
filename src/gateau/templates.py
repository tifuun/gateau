"""!
@file
File containing templates for input dictionaries.
"""

##
# Template for an instrument dictionary, to be passed to the interface object.
# 
# The current template is very much geared towards MKID spectrometers, but it can of course be tuned to other instrument types as well.
instrument = {
        "f0_ch"         : "Lowest frequency in filterbank in GHz (float or array). If an array is passed, 'nf_ch' is ignored and the filters are evaluated at the frequencies in the array",
        "nf_ch"         : "Number of frequencies in filterbank (int).",
        "R"             : "Resolving power f / df (int).",
        "f_sample"      : "Readout frequency in Hertz (float).",
        "box_eq"        : "If True, eta_filt is height of equivalent box filter. If False, eta_filt is peak height. Default is True.",
        "order"         : "Order of Lorentzian filterbank. Raises Lorentzian to given power, but keeps height same as normal Lorentzian. Defaults to 1.",
        "material"      : "Material(s) of MKID (string). If MKID is hybrid, for example Al and NbTiN, enter as: 'Al_NbTiN'."
        }

##
# Template for a telescope dictionary.
#
# Note that the sky chopping pattern defaults to no chopping. 
# If you want chopping, do not forget to set the 'chop_mode' parameter.
# If you do not want chopping, do not set this parameter.
# There are two aperture efficiencies, one for ON pos, and one for OFF.
# If only one is set, the other one will be set to the same value.
# Note that, when scantype = 'point', all fields beyond this field are set to zero.
telescope = {
        "eta_ap"     : "Aperture efficiency of telescope. If a single number is given, assume same aperture efficiency across entire frequency range.",
        "s_rms"         : "Surface rms roughness, in micrometer for Ruze efficiency. Leaving empty means no surface efficiency in calculation.",
        "az_scan"     : "Azimuth angles for scan, in arcesconds.",
        "el_scan"     : "Elevation angles for scan, in arcesconds.",
        }

##
# Template for atmosphere input dictionary.
atmosphere = {
        "Tatm"          : "Temperature of atmosphere in Kelvin.",
        "filename"      : "Name of file containing ARIS screen.",
        "path"          : "Path to ARIS file",
        "dx"            : "Gridsize of ARIS screen along x-axis in meters.",
        "dy"            : "Gridsize of ARIS screen along y-axis in meters.",
        "h_column"      : "Reference height of atmospheric column in meters.",
        "v_wind"        : "Windspeed in meters per second.",
        }
##
# Template for astronomical sources.
#
# The first part is common to all sources.
# After that, for each source the particular parts are mentioned.
Source = {
        "az_src"            : "Azimuthal range of source in arcseconds",
        "el_src"            : "Elevation range of source in arcseconds",
        "f_src"             : "Array with source frequencies in GHz, must be regular, i.e. uniform stepsize.",
        "I_nu"              : "Source cube, in azimuth, elevation, and frequencies. Units in SI, i.e. W / m**2 / sr / Hz.",
        }

