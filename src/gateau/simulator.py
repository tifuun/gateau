import math
import os
import shutil
import copy 
import sys
import numpy as np
from pathlib import Path

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

    Attributes:
        <type>Dict      :   Storage for input dictionaries. 
                            These are evaluated versions of the user input, containing references to actual data.
        clog_mgr        :   Logger manager. Top-level wrapper for actual logger.
                            Manager handles meta information, such as pwd.
        clog            :   Custom logger for communicating information, warnings, and errors to user.
        outPath         :   Path to directory where to store simulation output. Defaults to current working directory.
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

    outPath = os.getcwd() # Output path defaults to cwd    
    
    c = 2.99792458e8

    def __init__(self, 
                 verbose: bool = True, 
                 outPath: str = None) -> None:
        """!
        Create an interface object for gateau.

        @param verbose Set verbosity of logger. If False, will not log any output to screen.
            If True, will log simulation information, warnings, and errors to screen. 
            Default is True.
        @param outPath Path to where the interface object will write output. 
            Default is the current working directory.
        """

        self.outPath = outPath if outPath is not None else self.outPath
        if not verbose:
            self.clog.setLevel(logging.CRITICAL)
    
    def setLoggingVerbosity(self, 
                            verbose: bool = True) -> None:
        """!
        Set or change the verbosity of the gateau logger.
        
        @param verbose Give verbose output. Default is True.

        @ingroup settersgetters
        """

        if not verbose:
            self.clog.setLevel(logging.CRITICAL)
        
        else:
            self.clog.setLevel(logging.INFO)

    def setOutPath(self, 
                   outPath: str) -> None:
        """!
        Set or change path to output directory.

        @param outPath Path to output directory.

        @ingroup settersgetters
        """

        self.outPath = outPath

    def set_gateau_dict(self, 
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
        
        @ingroup settersgetters
        """
        
        setattr(self, label_dict, copy.deepcopy(input_dict))
        errlist = check_func(getattr(self, label_dict))
        
        if errlist:
            errstr = f"Errors encountered in {label_dict} dictionary in fields :{errlist}."
            raise FieldError(errstr)

    def initialise(self, 
                   t_obs: float, 
                   az0: float,
                   el0: float,
                   scan_func: Union[Callable, list[Callable]],
                   instrument_dict: dict[str, any],
                   telescope_dict: dict[str, any],
                   atmosphere_dict: dict[str, any],
                   source_dict: dict[str, any],
                   cascade_list: Union[list[dict[str, any]], str],
                   cascade_yaml: str = "cascade.yaml",
                   return_full: bool = False) -> Union[None, dict[str, any]]:
        """!
        Initialise a gateau setup. 


        @ingroup initialise
        """


        self.set_gateau_dict(instrument_dict, gcheck.checkInstrumentDict, "instrument")
        self.set_gateau_dict(telescope_dict, gcheck.checkTelescopeDict, "telescope")
        self.set_gateau_dict(atmosphere_dict, gcheck.checkAtmosphereDict, "atmosphere")
        self.set_gateau_dict(source_dict, gcheck.checkSourceDict, "source")

        if isinstance(cascade_list, str):
            cascade_list = gcascade.read_from_folder(cascade_list, cascade_yaml) 

        eta_cascade, psd_cascade, eta_ap, psd_cmb = gcascade.get_cascade(cascade_list, self.source["f_src"])

        eta_stage = np.array([x for arr in eta_cascade for x in arr])
        psd_stage = np.array([x for arr in psd_cascade for x in arr])

        self.cascade = {
                "eta_stage"     : eta_stage,
                "psd_stage"     : psd_stage,
                "num_stage"     : len(eta_cascade) - 1,
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

        if isinstance(scan_func, list):
            az_scan, el_scan = scan_func[0](times_array, az0, el0)
            self.telescope["az_scan_center"] = az_scan
            self.telescope["el_scan_center"] = el_scan
            for s_f in scan_func[1:]:
                az_scan, el_scan = s_f(times_array, az_scan, el_scan)
        
        else:
            az_scan_center, el_scan_center = scan_func(times_array, az0, el0)
            self.telescope["az_scan_center"] = az_scan_center
            self.telescope["el_scan_center"] = el_scan_center
            
            az_scan, el_scan = scan_func(times_array, az0, el0)

        self.telescope["az_scan"] = az_scan
        self.telescope["el_scan"] = el_scan
        
        f0_ch = self.instrument["f0_ch"]
        
        #### INITIALISING INSTRUMENT PARAMETERS ####
        if isinstance(f0_ch, np.ndarray):
            self.instrument["nf_ch"] = f0_ch.size
            self.instrument["f_ch_arr"] = f0_ch

        elif self.instrument.get("R") and self.instrument.get("nf_ch"):
            # R and number of channels given -> fl_ch unknowm
            idx_ch_arr = np.arange(self.instrument["nf_ch"])
            self.instrument["f_ch_arr"] = f0_ch * (1 + 1 / self.instrument["R"])**idx_ch_arr
        
        elif self.instrument.get("R") and self.instrument.get("fl_ch"):
            # R and fl_ch given -> number of channels unknown
            self.instrument["nf_ch"] = np.ceil(np.emath.logn(1 + 1/self.instrument["R"], self.instrument["fl_ch"]/self.instrument["f0_ch"])).astype(int)
            idx_ch_arr = np.arange(self.instrument["nf_ch"])
            self.instrument["f_ch_arr"] = f0_ch * (1 + 1 / self.instrument["R"])**idx_ch_arr
        
        elif self.instrument.get("nf_ch") and self.instrument.get("fl_ch"):
            # number of channels and fl_ch given -> filter Q unknown
            self.instrument["R"] = 1 / ((self.instrument["fl_ch"]/self.instrument["f0_ch"])**(1/(self.instrument["nf_ch"] - 1)) - 1)
            idx_ch_arr = np.arange(self.instrument["nf_ch"])
            self.instrument["f_ch_arr"] = f0_ch * (1 + 1 / self.instrument["R"])**idx_ch_arr
                
        if self.instrument["single_line"]:
            self.instrument["filterbank"] = gifu.generate_filterbank(self.instrument, self.source)
        else:
            self.instrument["filterbank"] = gifu.generate_filterbank_independent(self.instrument, self.source)

        # Now normalize to unit average peak height
        max_peak_mean = np.nanmean(
            np.nanmax(
                self.instrument["filterbank"], 
                axis=-1
                )
            )

        self.instrument["filterbank"] /= max_peak_mean

        if self.instrument["use_onef"]:
            if isinstance(self.instrument["onef_level"], float) or isinstance(self.instrument["onef_level"], int):
                self.instrument["onef_level"] *= np.ones(self.instrument["nf_ch"])
            if isinstance(self.instrument["onef_conv"], float) or isinstance(self.instrument["onef_conv"], int):
                self.instrument["onef_conv"] *= np.ones(self.instrument["nf_ch"])
            if isinstance(self.instrument["onef_alpha"], float) or isinstance(self.instrument["onef_alpha"], int):
                self.instrument["onef_alpha"] *= np.ones(self.instrument["nf_ch"])

        else:
            self.instrument["onef_level"] = np.zeros(self.instrument["nf_ch"])
            self.instrument["onef_conv"] = np.zeros(self.instrument["nf_ch"])
            self.instrument["onef_alpha"] = np.zeros(self.instrument["nf_ch"])
        
        if self.instrument.get("pointings") is None:
            if self.instrument.get("spacing") is None or self.instrument.get("radius") is None:
                self.instrument["pointings"] = np.zeros(1), np.zeros(1)
                self.instrument["n_spax"] = 1
            
            else:
                self.instrument["pointings"] = gifu.generate_fpa_pointings(self.instrument) 
                self.instrument["n_spax"] = self.instrument["pointings"][0].size
        

        #### INITIALISING TELESCOPE PARAMETERS ####
        self.telescope["eta_ruze"] = np.ones(self.source["f_src"].size)
        if isinstance(self.telescope.get("eta_taper"), float):
            self.telescope["eta_taper"] *= np.ones(self.source["f_src"].size)
        
        if self.telescope.get("s_rms") is not None:
            self.telescope["s_rms"] *= 1e-6 # Convert um to m

            eta_surf = np.exp(-(4 * np.pi * self.telescope["s_rms"] * self.source["f_src"] / self.c)**2)

            self.telescope["eta_ruze"] *= eta_surf 

        self.telescope["eta_illum"] = self.telescope["eta_taper"] * self.telescope["eta_ruze"]

        if return_full:
            eta_illum = copy.deepcopy(self.telescope["eta_illum"])
            eta_ap *= eta_illum

            for i, eta_cascade_stage in enumerate(eta_cascade):
                if not i:
                    eta_tot = eta_cascade_stage
                else:
                    eta_tot *= eta_cascade_stage

            eta_atm = get_eta_atm(self.source["f_src"],
                                  np.mean(self.atmosphere["PWV0"]),
                                  np.mean(el_scan))

            eta_ap_chan = gcascade.average_over_filterbank(eta_ap, 
                                                            self.instrument["filterbank"],
                                                            norm = True)
            
            eta_atm_chan = gcascade.average_over_filterbank(eta_atm, 
                                                             self.instrument["filterbank"],
                                                             norm = True)
            
            eta_tot_chan = gcascade.average_over_filterbank(eta_tot, 
                                                             self.instrument["filterbank"],
                                                             norm = True)
            
            out_dict = {
                    "eta_ap"     : eta_ap_chan,
                    "eta_atm"    : eta_atm_chan,
                    "eta_tot"    : eta_tot_chan,
                    "filterbank" : copy.deepcopy(self.instrument["filterbank"]),
                    }

            return out_dict 
        
    def run(self, 
            verbosity: int = 1, 
            outname: str = "out", 
            overwrite: bool = False,
            outscale: str = "Tb",
            seed: int = 0) -> None:
        """!
        Run a gateau simulation.

        This is the main routine of gateau and should be called after filling all dictionaries and running the self.initialise() method.
        
        @param t_obs Total observation time in seconds.
        @param verbosity Level of verbosity for simulation.
            0           : no extra output w.r.t. logger.
            1 (default) : show execution times of important routines.
        @param outname Name ofirectory to store output in. Directory will be placed in outPath.
        @param overwrite Whether to overwrite existing output directories. 
            If False (default), a prompt will appear to either overwrite or specify new directory.
        @param outscale Store output in brightness temperature [K] or power [W].
            Accepts "Tb" or "P". Defaults to "Tb".
        @param seed Seed for photon and 1/f noise generation. 
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

        path.mkdir(parents=True, exist_ok=True)

        # Check if enough HDD is available in outpath for this simulation
        n_bytes_required = ((self.instrument["nf_ch"] + 2)*self.instrument["n_spax"] + 1) * self.n_times * 4
        if (n_bytes_free := int(MEMFRAC * shutil.disk_usage(self.outPath).free)) < n_bytes_required:
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
