import math
import os
import shutil
import time
import copy 
import sys
import numpy as np

import gateau.ifu as gifu
import gateau.input_checker as gcheck
import gateau.bindings as gbind
import gateau.cascade as gcascade

import psutil
import logging
from gateau.custom_logger import CustomLogger
from gateau.utilities import get_eta_atm

from collections.abc import Callable
from typing import Union

logging.getLogger(__name__)

MEMFRAC = 0.5
MEMBUFF = MEMFRAC * psutil.virtual_memory().total
T_OBS_BUFF = 0.8

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
                   scan_func: Union[Callable, list[Callable]],
                   instrument_dict: dict[str, any],
                   telescope_dict: dict[str, any],
                   atmosphere_dict: dict[str, any],
                   source_dict: dict[str, any],
                   cascade_list: list[dict[str, any]]) -> tuple[np.ndarray, 
                                                                np.ndarray, 
                                                                np.ndarray, 
                                                                np.ndarray]:
        """!
        Initialise a gateau setup. 

        @param use_ARIS Whether to load an ARIS screen or not. 
            Some functions of gateau do not require an ARIS screen to be loaded. 
            Setting this parameter to False could reduce total memory footprint in these cases.
            Default is True (load the ARIS screen).
        @param number Number of ARIS chunks to concatenate and load into memory.
        @param start ARIS chunk to start with. 
        
        @returns Three arrays: 
                - total transmission efficiency of cascade for an astronomical source, averaged over filterbank.
                - azimuth angle over total scan.
                - elevation angle over total scan.

        @ingroup initialise
        """


        self.set_gateau_dict(instrument_dict, gcheck.checkInstrumentDict, "instrument")
        self.set_gateau_dict(telescope_dict, gcheck.checkTelescopeDict, "telescope")
        self.set_gateau_dict(atmosphere_dict, gcheck.checkAtmosphereDict, "atmosphere")
        self.set_gateau_dict(source_dict, gcheck.checkSourceDict, "source")

        eta_cascade, psd_cascade = gcascade.get_cascade(cascade_list, self.source["f_src"])

        eta_stage = np.array([x for arr in eta_cascade for x in arr])
        psd_stage = np.array([x for arr in psd_cascade for x in arr])

        self.cascade = {
                "eta_stage"     : eta_stage,
                "psd_stage"     : psd_stage,
                "num_stage"     : len(eta_cascade) - 1
                }

        #### END SETUP INITIALISATION ####
        self.initialisedSetup = True
        
        #### INITIALISING OBSERVATION PARAMETERS ####
        # Calculate number of time evaluations
        # Note that, in order to simplify TLS noise calculations, we make nTimes even
        self.nTimes = math.ceil(t_obs * self.instrument["f_sample"])

        if self.nTimes % 2 == 1:
            self.nTimes -= 1
        
        times_array = np.arange(0, self.nTimes / self.instrument["f_sample"], 1 / self.instrument["f_sample"])

        if isinstance(scan_func, list):
            az_scan, el_scan = scan_func[0](times_array)
            for s_f in scan_func[1:]:
                az_scan, el_scan = s_f(times_array, az_scan, el_scan)
        
        else:
            az_scan, el_scan = scan_func(times_array)

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
                
        self.instrument["filterbank"] = gifu.generateFilterbankFromR(self.instrument, self.source)
        
        if self.instrument.get("pointings") is None:
            if self.instrument.get("spacing") is None or self.instrument.get("radius") is None:
                self.instrument["pointings"] = np.zeros(1), np.zeros(1)
            
            else:
                self.instrument["pointings"] = gifu.generate_fpa_pointings(self.instrument) 

        #### INITIALISING TELESCOPE PARAMETERS ####
        if isinstance(self.telescope.get("eta_ap"), float):
            self.telescope["eta_ap"] *= np.ones(self.source["f_src"].size)
        
        if self.telescope["s_rms"] is not None:
            self.telescope["s_rms"] *= 1e-6 # Convert um to m

            eta_surf = np.exp(-(4 * np.pi * self.telescope["s_rms"] * self.source["f_src"] / self.c)**2)

            self.telescope["eta_ap"] *= eta_surf 

        # Some handy returns
        eta_total = self.telescope["eta_ap"]
        for eta in eta_cascade:
            eta_total *= eta

        eta_atm = get_eta_atm(self.source["f_src"],
                              self.atmosphere["PWV0"],
                              np.mean(el_scan))

        return eta_total, eta_atm, az_scan, el_scan
        
    def run(self, 
            verbosity: int = 1, 
            outname: str = "out", 
            overwrite: bool = False) -> None:
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
        """

        if not self.initialisedSetup:
            self.clog.error("The initialise method MUST be called before running a simulation!")
            raise InitialError
            sys.exit()
        t_range = 1 / self.instrument["f_sample"] * np.arange(self.nTimes)

        outpath = os.path.join(self.outPath, outname)

        outpath_succes = False
        while not outpath_succes:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                outpath_succes = True

            elif overwrite:
                shutil.rmtree(outpath)
                os.makedirs(outpath)
                outpath_succes = True

            elif not overwrite:
                self.clog.warning(f"Output path {outpath} exists! Overwrite or change path?")
                choice = input("\033[93mOverwrite (y/n)? > ").lower()
                if choice == "y" or choice == "":
                    shutil.rmtree(outpath)
                else:
                    outpath = input("\033[93mSpecify new output path: > ")

        # Create folders for each spaxel
        for idx_spax in range(self.instrument.get("pointings")[0].size):
            os.makedirs(os.path.join(outpath, str(idx_spax)))
        self.clog.info("\033[1;32m*** STARTING gateau SIMULATION ***")
        
        start = time.time()

        gbind.run_gateau(self.instrument, 
                         self.telescope, 
                         self.atmosphere, 
                         self.source,
                         self.cascade,
                         self.nTimes, 
                         outpath)
        
        end = time.time()        
        
        self.clog.info("\033[1;32m*** FINISHED gateau SIMULATION ***")

