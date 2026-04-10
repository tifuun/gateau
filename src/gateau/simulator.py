"""!
@file simulator.py
@brief Interface file for the gateau simulator. 
    This file contains the public methods necessary for initialising and running gateau simulations. 
"""

import math
import os
import shutil
import copy 
import sys
import numpy as np
from pathlib import Path
from scipy.constants import c

import gateau.ifu as gifu
import gateau.input_checker as gcheck
import gateau.bindings as gbind
import gateau.cascade as gcascade

import logging
from gateau.custom_logger import CustomLogger
from gateau.atmosphere_utils import get_eta_atm

from collections.abc import Callable
from typing import Union

logging.getLogger(__name__)

MEMFRAC = 0.8
LIM_ATM_LO = 0.1
LIM_ATM_HI = 5.5

class FieldError(Exception):
    """!
    Field error. Raised when a required field is not specified in an input dictionary. 
    """
    pass

class InitialError(Exception):
    """!
    Initial error. Raised when attempting to run a simulation without submitting all required dictionaries. 
    """
    pass

class simulator(object):
    """!
    Interface for gateau simulations
    """

    atmosphere      = None
    telescope       = None
    instrument      = None
    observation     = None
    source          = None

    cascadeList         = None    

    initialisedSetup    = False

    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()
    
    c = 2.99792458e8

    def __init__(self, 
                 verbose: bool = True) -> None:
        """!
        Create an interface object for gateau.

        @param verbose Set verbosity of logger. If False, will not log any output to screen.
            If True, will log simulation information, warnings, and errors to screen. 
            Default is True.
        """

        if not verbose:
            self.clog.setLevel(logging.CRITICAL)
    
    def setLoggingVerbosity(self, 
                            verbose: bool) -> None:
        """!
        Set or change the verbosity of the gateau logger.

        @ingroup public_API_settersgetters
        
        @param verbose Set verbosity of logger. If False, will not log any output to screen.
            If True, will log simulation information, warnings, and errors to screen. 
        """

        if not verbose:
            self.clog.setLevel(logging.CRITICAL)
        
        else:
            self.clog.setLevel(logging.INFO)

    def initialise(self, 
                   t_obs: float, 
                   az0: float,
                   el0: float,
                   scan_func: Callable,
                   instrument_dict: dict[str, any],
                   telescope_dict: dict[str, any],
                   atmosphere_dict: dict[str, any],
                   source_dict: dict[str, any],
                   cascade_list: Union[list[dict[str, any]], str],
                   cascade_yaml: str = "cascade.yaml",
                   return_full: bool = False) -> Union[None, dict[str, any]]:
        """!
        Initialise a gateau setup. 
        THis function needs to be called before running a simulation.
        Here, a lot of intermediary user-supplied quantities are converted into quantities used by gateau.
    
        @ingroup public_api_simulator

        @param t_obs Total observation time for simulation, in seconds.
        @param az0 Central azimuth value for the scan pattern, in degrees.
        @param el0 Central elevation value for the scan pattern, in degrees.
        @param scan_func Function handle of the function defining the scan pattern. 
            First argument must be a Numpy array consisting of timestamps.
            Second and third argument must be scalars or Numpy arrays containing central azimuth and elevation values, respectively.
        @param instrument_dict Dictionary containing instrument specification.
        @param telescope_dict Dictionary containing telescope specification.
        @param atmosphere_dict Dictionary containing atmosphere specification.
        @param source_dict Dictionary containing source specification.
        @param cascade_list List containing the cascade to be used. 
            Can also be a string containg the path to the folder containing a cascade .yaml file.
        @param cascade_yaml Name of .yaml file containing cascade.
            Only used if 'cascade_list' is a string containing a folder with a cascade .yaml.
            Defaults to 'cascade.yaml'.
        @param return_full Boolean determining whether extra output is returned.
            This extra output might be useful when you want to process the actual gateau output further.
            Defaults to False.

        @returns Dictionary containing the aperture efficiency and atmospheric transmission.
            The latter is evaluated using the PWV0 supplied in the atmosphere dictionary.
            Both quantities are averaged over the spectral shape of each channel.
            The dictionary is only returned when 'return_full' is True.
        """
        self._set_gateau_dict(instrument_dict, gcheck.checkInstrumentDict, "instrument")
        self._set_gateau_dict(telescope_dict, gcheck.checkTelescopeDict, "telescope")
        self._set_gateau_dict(atmosphere_dict, gcheck.checkAtmosphereDict, "atmosphere")
        self._set_gateau_dict(source_dict, gcheck.checkSourceDict, "source")

        if isinstance(cascade_list, str):
            cascade_list = gcascade.read_from_folder(cascade_list, cascade_yaml) 

        eta_cascade, psd_cascade, eta_ap, psd_cmb = gcascade.get_cascade(cascade_list, self.source["f_src"])

        eta_stage = np.array([x for arr in eta_cascade for x in arr])
        psd_stage = np.array([x for arr in psd_cascade for x in arr])

        self.cascade = {
                "eta_stage"     : eta_stage,
                "psd_stage"     : psd_stage,
                "num_stage"     : len(eta_cascade),
                "psd_cmb"       : psd_cmb
                }

        #### END SETUP INITIALISATION ####
        self.initialisedSetup = True
        
        #### INITIALISING OBSERVATION PARAMETERS ####
        # Calculate number of time evaluations
        # Note that, in order to simplify TLS noise calculations, we make n_times even
        self.n_times = math.ceil(t_obs * self.instrument["f_sample"])

        if self.n_times % 2 == 1:
            self.n_times -= 1

        times_array = np.arange(0, self.n_times / self.instrument["f_sample"], 1 / self.instrument["f_sample"])
        
        # Checking observation time against available time
        atm_meta = np.loadtxt(os.path.join(self.atmosphere["path"], "prepd", "atm_meta.datp"))
        t_available = atm_meta[0] * (atm_meta[1] - 2 * atm_meta[2]) * self.atmosphere["dx"] / (self.atmosphere["v_wind"] + np.finfo(float).eps)

        if t_available < t_obs:
            self.clog.warning(f"Requested observation time of {t_obs} s exceeds available time of {t_available} s in ARIS screens. Reducing requested time to available time.")
            choice = input("\033[93mProceed (y/n)? > ").lower()
            if choice == "y" or choice == "":
                t_obs = t_available
            else:
                exit()

        # Checking if start/end PWV will not be lower/higher than ATM tabulated PWV values
        # We take into account min/max of the ARIS screens
        avg_PWV = np.nanmean(self.atmosphere["PWV0"])

        if (avg_PWV + atm_meta[3]) < LIM_ATM_LO:

            self.clog.warning(f"Average PWV of {avg_PWV} mm will likely go below tabulated limit of {LIM_ATM_LO} with the supplied screens.")
            choice = input("\033[93mProceed (y/n)? > ").lower()
            if choice == "y" or choice == "":
                pass
            else:
                exit()

        if (avg_PWV + atm_meta[4]) > LIM_ATM_HI:

            self.clog.warning(f"Average PWV of {avg_PWV} mm will likely go above tabulated limit of {LIM_ATM_HI} with the supplied screens.")
            choice = input("\033[93mProceed (y/n)? > ").lower()
            if choice == "y" or choice == "":
                pass
            else:
                exit()

        # We also convert the average pwv tuple in the atmosphere dict to a starting pwv and a slope
        self.atmosphere["PWV_slope"] = (self.atmosphere["PWV0"][1] - self.atmosphere["PWV0"][0]) / times_array[-1]

        az_scan_center, el_scan_center = scan_func(times_array, az0, el0)
        self.telescope["az_scan_center"] = az_scan_center
        self.telescope["el_scan_center"] = el_scan_center
        
        az_scan, el_scan = scan_func(times_array, az0, el0)

        self.telescope["az_scan"] = az_scan
        self.telescope["el_scan"] = el_scan
        
        #### INITIALISING INSTRUMENT PARAMETERS ####
        # First, check if channel frequencies are given.
        R_inst = self.instrument.get("R")
        nf_ch = self.instrument.get("nf_ch")
        fmin_ch = self.instrument.get("fmin_ch")
        fmax_ch = self.instrument.get("fmax_ch")

        if (f_ch := self.instrument.get("f_ch")) is None:
            # R, nf_ch, and fmin_ch given -> fmax_ch unknowm
            if (R_inst := self.instrument.get("R")) is not None and \
                (nf_ch := self.instrument.get("nf_ch")) is not None and \
                (fmin_ch := self.instrument.get("fmin_ch")) is not None and\
                (fmax_ch := self.instrument.get("fmax_ch")) is None:

                idx_ch_arr = np.arange(nf_ch)
                self.instrument["f_ch"] = fmin_ch * (1 + 1 / R_inst)**idx_ch_arr
            
            # R, nf_ch, and fmax_ch given -> fmin_ch unknown
            elif R_inst is not None and \
                    nf_ch is not None and \
                    fmax_ch is not None and \
                    fmin_ch is None:

                idx_ch_arr = np.arange(nf_ch)
                fmin_ch = fmax_ch * (1 + 1 / R_inst)**(1 - nf_ch)
                self.instrument["f_ch"] = fmin_ch * (1 + 1 / R_inst)**idx_ch_arr

            # R, fmin_ch, and fmax_ch given -> nf_ch unknown
            elif R_inst is not None and \
                    fmin_ch is not None and \
                    fmax_ch is not None and \
                    nf_ch is None:

                nf_ch = np.ceil(np.emath.logn(1 + 1 / R_inst, fmax_ch / fmin_ch)).astype(int)
                idx_ch_arr = np.arange(nf_ch)
                self.instrument["f_ch"] = fmin_ch * (1 + 1 / R_inst)**idx_ch_arr
            
            # nf_ch, fmin_ch, and fmax_ch given -> R unknown
            elif  nf_ch is not None and \
                    fmin_ch is not None and \
                    fmax_ch is not None and \
                    R_inst is None:
                R_inst = 1 / ((fmax_ch / fmin_ch)**(1/(nf_ch - 1)) - 1)
                idx_ch_arr = np.arange(nf_ch)
                self.instrument["f_ch"] = fmin_ch * (1 + 1 / R_inst)**idx_ch_arr

            self.instrument["nf_ch"] = nf_ch
            self.instrument["R"] = R_inst

        else:
            self.instrument["nf_ch"] = self.instrument["f_ch"].size

        if self.instrument["use_filterbank"]:
            self.instrument["transmission"] = gifu.generate_transmission(self.instrument, self.source)
        elif self.instrument.get("transmission") is None:
            self.instrument["transmission"] = gifu.generate_transmission_independent(self.instrument, self.source)
        else:
            self.instrument["transmission"] = gcascade.sizer(
                    self.instrument["transmission"][0], 
                    self.source["f_src"], 
                    self.instrument["transmission"][1],
                    axis = 0
                    )

        if self.instrument["use_pink"]:
            if isinstance(self.instrument["pink_level"], float) or isinstance(self.instrument["pink_level"], int):
                self.instrument["pink_level"] *= np.ones(self.instrument["nf_ch"])
            if isinstance(self.instrument["pink_alpha"], float) or isinstance(self.instrument["pink_alpha"], int):
                self.instrument["pink_alpha"] *= np.ones(self.instrument["nf_ch"])

        else:
            self.instrument["pink_level"] = np.zeros(self.instrument["nf_ch"])
            self.instrument["pink_alpha"] = np.zeros(self.instrument["nf_ch"])
        
        if self.instrument.get("pointings") is None:
            if self.instrument.get("spacing") is None or self.instrument.get("radius") is None:
                self.instrument["pointings"] = np.zeros(1), np.zeros(1)
            
            else:
                self.instrument["pointings"] = gifu.generate_fpa_pointings(self.instrument) 

        # In any case, want to get number of spaxels
        self.instrument["n_spax"] = self.instrument["pointings"][0].size

        #### INITIALISING TELESCOPE PARAMETERS ####
        self.telescope["eta_ruze"] = np.ones((f_src := self.source["f_src"]).size)
        if isinstance(eta_t := self.telescope.get("eta_taper"), float):
            self.telescope["eta_taper"] *= np.ones(f_src.size)
        
        elif isinstance(eta_t, tuple):
            self.telescope["eta_taper"] = gcascade.sizer(eta_t[0], f_src, eta_t[1])
        
        if self.telescope.get("s_rms") is not None:
            self.telescope["s_rms"] *= 1e-6 # Convert um to m

            eta_surf = np.exp(-(4 * np.pi * self.telescope["s_rms"] * self.source["f_src"] / c)**2)

            self.telescope["eta_ruze"] *= eta_surf 

        self.telescope["eta_illum"] = self.telescope["eta_taper"] * self.telescope["eta_ruze"]

        if return_full:
            eta_illum = copy.deepcopy(self.telescope["eta_illum"])
            eta_ap *= eta_illum

            eta_atm = get_eta_atm(self.source["f_src"],
                                  np.mean(self.atmosphere["PWV0"]),
                                  np.mean(el_scan))
            
            return {
                    "eta_ap"     : self._average_over_transmission(eta_ap),
                    "eta_atm"    : self._average_over_transmission(eta_atm),
                    }
        
    def run(self, 
            outname: Union[str, Path] = "out", 
            overwrite: bool = False,
            outscale: str = "Tb",
            seed: int = 0) -> None:
        """!
        Run a gateau simulation.
        This is the main routine of gateau and should be called after filling all dictionaries and running the 'initialise' method.
    
        @ingroup public_api_simulator
        
        @param outname Name of output hdf5 file. If a path, will place output in the path.
            Defaults to 'out', which will place the output in 'out.hdf5' in your working directory.
        @param overwrite Whether to overwrite existing output directories. 
            If False (default), a prompt will appear to either overwrite or terminate simulation.
        @param outscale Store output in brightness temperature [K] or power [W].
            Accepts "Tb" or "P". Defaults to "Tb".
        @param seed Seed for photon and pink noise generation. 
            Defaults to 0, which will internally be converted to a random seed using the current time.
        """

        if not self.initialisedSetup:
            self.clog.error("The initialise method MUST be called before running a simulation!")
            raise InitialError
            sys.exit()

        if isinstance(outname, str):
            outname = Path(outname)

        outname = outname.with_suffix(".h5")
        path = outname.parent

        if outname.is_file() and not overwrite:
            self.clog.warning(f"File {outname.name} already exists in {path.resolve()}.")
            choice = input("\033[93mProceed (y/n)? > ").lower()
            if choice == "y" or choice == "":
                pass
            else:
                exit()

        path.mkdir(parents=True, exist_ok=True)

        # Check if enough HDD is available in outpath for this simulation
        n_bytes_required = ((self.instrument["nf_ch"] + 2)*self.instrument["n_spax"] + 1) * self.n_times * 4
        if (n_bytes_free := int(MEMFRAC * shutil.disk_usage(path).free)) < n_bytes_required:
            self.clog.warning(f"Required disk space of {n_bytes_required} bytes exceeds available buffer of {n_bytes_free} bytes.")
            choice = input("\033[93mProceed (y/n)? > ").lower()
            if choice == "y" or choice == "":
                pass
            else:
                exit()

        self.clog.info("\033[1;32m*** STARTING gateau SIMULATION ***")
        
        gbind.run_gateau(self.instrument, 
                            self.telescope, 
                        self.atmosphere, 
                             self.source,
                        self.cascade,
                             self.n_times, 
                     outname,
                                       outscale,
                  seed)

        self.clog.info("\033[1;32m*** FINISHED gateau SIMULATION ***")
    
    def _set_gateau_dict(self, 
                        input_dict: dict[str, any], 
                        check_func: Callable, 
                        label_dict: str) -> None:
        """!
        Pass an input dictionary and check it for correctness.

        @param input_dict Dictionary to be checked.
        @param check_func Function that should check correctness of input dictionary.
        @param self_dict Class attribute dictionary to which input dictionary should be copied.
        @param label_dict Label for input dictionary used for printing the error string, if needed.
        @param flag_set Class attribute flag that should be set to True upon acceptance of input dictionary.
        """
        
        setattr(self, label_dict, copy.deepcopy(input_dict))
        errlist = check_func(getattr(self, label_dict))
        
        if errlist:
            errstr = f"Errors encountered in {label_dict} dictionary in fields :{errlist}."
            raise FieldError(errstr)

    def _average_over_transmission(self, array_to_average: np.ndarray) -> np.ndarray:
        """!
        Average an array over the transmission curves.
        
        @param array_to_average Numpy array with size equal to size of source frequency array.

        @returns Array averaged over transmission curves.
        """
        sh_f = self.instrument["transmission"].shape
        assert array_to_average.size == sh_f[1]

        array_tiled = np.squeeze(array_to_average)[None,:] * self.instrument["transmission"]
        return np.nansum(array_tiled, axis=1) / np.nansum(self.instrument["transmission"], axis=1)
