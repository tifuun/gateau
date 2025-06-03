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
        "f0_src"        : "Lowest frequency in source in GHz (float).",
        "f1_src"        : "Largest frequency in source in GHz (float).",
        "nf_src"        : "Number of source frequency points (int).",
        "R"             : "Resolving power f / df (int).",
        "eta_inst"      : "Efficiency of entire chip, from antenna up to filterbank (float).",
        "f_sample"      : "Readout frequency in Hertz (float).",
        "eta_filt"      : "Filter efficiency (float or array of floats).",
        "box_eq"        : "If True, eta_filt is height of equivalent box filter. If False, eta_filt is peak height. Default is True.",
        "order"         : "Order of Lorentzian filterbank. Raises Lorentzian to given power, but keeps height same as normal Lorentzian. Defaults to 1."
        "material"      : "Material(s) of MKID (string). If MKID is hybrid, for example Al and NbTiN, enter as: 'Al_NbTiN'."
        "eta_misc"      : "Miscellaneous constant efficiency terms."
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
        "Dtel"          : "Diameter of telescope in meters.",
        "Ttel"          : "Temperature of telescope in Kelvin.",
        "Tgnd"          : "Temperature of ground around telescope in Kelvin.",
        "eta_ap_ON"     : "Aperture efficiency of telescope in chop ON, as function of instrument frequencies. If a single number is given, assume same aperture efficiency across entire frequency range.",
        "eta_ap_OFF"    : "Aperture efficiency of telescope in chop OFF, as function of instrument frequencies. If a single number is given, assume same aperture efficiency across entire frequency range.",
        "eta_mir"       : "Mirror efficiency of telescope.",
        "eta_fwd"       : "Front-to-back efficiency.",
        "s_rms"         : "Surface rms roughness, in micrometer for Ruze efficiency. Leaving empty means no surface efficiency in calculation."
        "chop_mode"     : "How to chop. Can choose 'none', 'direct', 'abba'."
        "freq_chop"     : "Chopping frequency in Hertz. If None, no chopping.",
        "dAz_chop"      : "Angular separation between chopping paths in arcseconds.",
        "scantype"      : "Type of scanning pattern. Can choose between 'point' or 'daisy'. Default is point.",
        "El0"           : "Elevation of telescope in degrees.",
        "Ax"            : "Amplitude of petal of Daisy scan, along Azimuth axis, in arcsec.",
        "Ay"            : "Amplitude of petal of Daisy scan, along Elevation axis, in arcsec.",
        "Axmin"         : "Amplitude of Daisy scan, along Azimuth axis, in arcsec.",
        "Aymin"         : "Amplitude of Daisy scan, along Elevation axis, in arcsec.",
        "wx"            : "Angular velocity of petal of Daisy scan, along Azimuth axis, in arcsec / sec.",
        "wy"            : "Angular velocity of petal of Daisy scan, along Elevation axis, in arcsec / sec.",
        "wxmin"         : "Angular velocity of Daisy scan, along Azimuth axis, in arcsec / sec.",
        "wymin"         : "Angular velocity of Daisy scan, along Elevation axis, in arcsec / sec.",
        "phix"          : "Phase of Daisy scan along Azimuth axis, in arcsec.",
        "phiy"          : "Phase of Daisy scan along Elevation axis, in arcsec.",
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
        "PWV0"          : "Mean PWV value in millimeters.",
        }
##
# Template for astronomical sources.
#
# The first part is common to all sources.
# After that, for each source the particular parts are mentioned.
Source = {
        "type"          : "Type of source ('SZ' or 'GalSpec').",
        "Az"            : "Azimuthal lower and upper limits of source map in degrees.",
        "El"            : "Elevation lower and upper limits of source map in degrees.",
        "nAz"           : "Number of Azimuth points.",
        "nEl"           : "Number of Elevation points.",
        # MockSZ specific
        "Te"            : "Electron temperature of cluster gas in Kev.",
        "ne0"           : "Central electron density in # per square centimeter.",
        "beta"          : "Isothermal-beta structure coefficient.",
        "v_pec"         : "Peculiar cluster velocity, relative to CMB, in kilometers per second.",
        "thetac"        : "Cluster core radius in arcsec.",
        "Da"            : "Angular diameter distance in megaparsec.",
        "freqs_src"     : "Range of frequencies over which to simulate source signal, in GHz.",
        # GalSpec specific
        "lum"           : "Luminosity in log(L_fir/L_sol).",
        "z"             : "redshift of galaxy.",
        "lwidth"        : "Linewidth of spectral lines in km/s.",
        "COlines"       : "Kamenetzky or Rosenberg.",
        "lines"         : "Bonato or Spinoglio.",
        "mollines"      : "T/F, add molecular lines."
        }

load_source = {
        "path"          : "Path to saved source datacube.",
        "filename"      : "Name of saved source.",
        }

##
# Template for a power-temperature (Watt-to-Kelvin, or W2K for short) dictionary.
# Because TiEMPO2 calculates in SI units of Watt, it is necessary to calculate the calibration table if one wants to convert to Kelvin scale.
W2K = {
        "nPWV"          : "Number of PWV values to use in calculation. The actual range is set by the eta_atm screen, generated using ATM.",
        "nThreads"      : "Number of CPU threads to use for calculation.",
        }
