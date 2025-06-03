import math
import os
import shutil
import time
import copy 
import sys
import numpy as np

import gateau.filterbank as gfilter
import gateau.input_checker as gcheck
import gateau.bindings as gbind

import psutil
import logging
from gateau.custom_logger import CustomLogger

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

class Interface(object):
    """!
    Interface for gateau simulations

    Attributes:
        __<type>Dict    :   Storage for input dictionaries. These dictionaries are raw copies of user input dicts.
                            When the input dictionary passes the input test, it will be copied to this dictionary.
        <type>Dict      :   Storage for input dictionaries. 
                            These are evaluated versions of the user input, containing references to actual data.
        <type>_set      :   Flags for signifying that a dictionary has been validly set. 
                            Will only be set to 'True' if the dictionary passes the input test.
        clog_mgr        :   Logger manager. Top-level wrapper for actual logger.
                            Manager handles meta information, such as pwd.
        clog            :   Custom logger for communicating information, warnings, and errors to user.
        outPath         :   Path to directory where to store simulation output. Defaults to current working directory.
    """

    __atmosphereDict    = None
    __telescopeDict     = None
    __instrumentDict    = None
    __observationDict   = None

    atmosphereDict      = None
    telescopeDict       = None
    instrumentDict      = None
    observationDict     = None
    sourceDict          = None

    atmosphere_set      = False
    telescope_set       = False
    instrument_set      = False
    observation_set     = False
    source_set          = False
    
    initialisedSetup    = False
    initialisedObserve  = False

    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()

    outPath = os.getcwd() # Output path defaults to cwd    
    
    c = 2.99792458e8

    def __init__(self, verbose=True, outPath=None):
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
    
    def setLoggingVerbosity(self, verbose=True):
        """!
        Set or change the verbosity of the gateau logger.
        
        @param verbose Give verbose output. Default is True.

        @ingroup settersgetters
        """

        if not verbose:
            self.clog.setLevel(logging.CRITICAL)
        
        else:
            self.clog.setLevel(logging.INFO)

    def setOutPath(self, outPath):
        """!
        Set or change path to output directory.

        @param outPath Path to output directory.

        @ingroup settersgetters
        """

        self.outPath = outPath

    def setTelescopeDict(self, telescopeDict):
        """!
        Set a telescope dictionary.

        @param telescopeDict A dictionary specifying the telescope for the simulation.

        @ingroup inputdicts
        """

        errlist = gcheck.checkTelescopeDict(telescopeDict)

        if not errlist:
            self.__telescopeDict = telescopeDict
            self.telescope_set = True

        else:
            errstr = f"Errors encountered in telescope dictionary in fields :{errlist}."
            raise FieldError(errstr)
    
    def setInstrumentDict(self, instrumentDict):
        """!
        Set an instrument dictionary.

        @param instrumentDict A dictionary specifying the instrument to simulate.

        @ingroup inputdicts
        """

        errlist = gcheck.checkInstrumentDict(instrumentDict)

        if not errlist:
            self.__instrumentDict = instrumentDict
            self.instrument_set = True

        else:
            errstr = f"Errors encountered in instrument dictionary in fields :{errlist}."
            raise FieldError(errstr)
        
    
    def setAtmosphereDict(self, atmosphereDict):
        """!
        Set an atmosphere dictionary.
        Note that this dictionary only specifies some parameters, such as windspeed, scale height, etc. Actual atmosphere screens need to be generated using ARIS. 
        Also, the directory containing the ARIS screens, together with the actual name of the screen files, is specified here.

        @param atmosphereDict A dictionary specifying the atmosphere to simulate.

        @ingroup inputdicts
        """

        errlist = gcheck.checkAtmosphereDict(atmosphereDict)

        if not errlist:
            self.__atmosphereDict = atmosphereDict
            self.atmosphere_set = True

        else:
            errstr = f"Errors encountered in atmosphere dictionary in fields :{errlist}."
            raise FieldError(errstr)

    # NUMBER PARAMETER IS TEMPORARY
    def initSetup(self, use_ARIS=True, number=1, start=0):
        """!
        Initialise a gateau setup. 

        The idea is that a gateau setup can be done sequentially, the first step being a setup of the terrestrial part.
        This means that the first setup should consist of everything on Earth: instrument, telescope, and atmosphere.
        Therefore, this is the first initialisation that should be performed.

        This function will check wether or not the instrument, telescope, and atmosphere are set.
        If not, an InitialError will be raised.

        @param use_ARIS Whether to load an ARIS screen or not. 
            Some functions of gateau do not require an ARIS screen to be loaded. 
            Setting this parameter to False could reduce total memory footprint in these cases.
            Default is True (load the ARIS screen).
        @param number Number of ARIS chunks to concatenate and load into memory.
        @param start ARIS chunk to start with. 
        
        @ingroup initialise
        """

        if not (self.instrument_set and 
                self.telescope_set and
                self.atmosphere_set):
            
            self.clog.error("an instrument, telescope, and atmosphere dictionary MUST be set before calling initSetup()")
            raise InitialError
            sys.exit()
       
        # Deepcopy raw dicts into proper input dicts, so to not have to interact with raw templates.
        self.instrumentDict = copy.deepcopy(self.__instrumentDict)
        self.telescopeDict = copy.deepcopy(self.__telescopeDict)
        self.atmosphereDict = copy.deepcopy(self.__atmosphereDict)
        
        #### END INITIALISATION ####
        self.initialisedSetup = True

    def initSource(self, source_cube, az_grid, el_grid, f_arr, pointing=None):
        """!
        Initialise a gateau source. 

        The idea is that a gateau setup can be done sequentially, the first step being a setup of the terrestrial part.
        This means that the second setup should consist of everything in space: the astronomical source.
        Therefore, this is the second initialisation that should be performed.

        @param pointing Pointing center of telescope, w.r.t. center of astronomical source, in arcseconds.
        
        @ingroup initialise
        """

        az_arr = az_grid[:,0]
        el_arr = el_grid[0,:]
        nf_src = f_arr.size

        self.instrumentDict["f0_src"] = f_arr[0]
        self.instrumentDict["f1_src"] = f_arr[-1]
        self.instrumentDict["f_src"] = f_arr
        self.instrumentDict["nf_src"] = nf_src

        pointing = np.zeros(2) if pointing is None else pointing

        f0_ch = self.instrumentDict["f0_ch"]
        
        #### INITIALISING INSTRUMENT PARAMETERS ####
        # Generate filterbank
        if isinstance(f0_ch, np.ndarray):
            self.instrumentDict["nf_ch"] = f0_ch.size
            self.instrumentDict["f_ch_arr"] = f0_ch

        elif self.instrumentDict.get("R") and self.instrumentDict.get("nf_ch"):
            # R and number of channels given -> fl_ch unknowm
            idx_ch_arr = np.arange(self.instrumentDict["nf_ch"])
            self.instrumentDict["f_ch_arr"] = f0_ch * (1 + 1 / self.instrumentDict["R"])**idx_ch_arr
        
        elif self.instrumentDict.get("R") and self.instrumentDict.get("fl_ch"):
            # R and fl_ch given -> number of channels unknown
            self.instrumentDict["nf_ch"] = np.ceil(np.emath.logn(1 + 1/self.instrumentDict["R"], self.instrumentDict["fl_ch"]/self.instrumentDict["f0_ch"])).astype(int)
            idx_ch_arr = np.arange(self.instrumentDict["nf_ch"])
            self.instrumentDict["f_ch_arr"] = f0_ch * (1 + 1 / self.instrumentDict["R"])**idx_ch_arr
        self.instrumentDict["eta_filt"] *= np.ones(self.instrumentDict.get("nf_ch"))
                
        self.instrumentDict["filterbank"] = gfilter.generateFilterbankFromR(self.instrumentDict)
        #import matplotlib.pyplot as plt

        #plt.plot(self.instrumentDict["filterbank"].T)
        #plt.show()
        
        self.sourceDict = {"I_nu"   : source_cube,
                           "Az_src" : az_arr / 3600,
                           "El_src" : el_arr / 3600}

        #### INITIALISING TELESCOPE PARAMETERS ####
        if isinstance(self.telescopeDict.get("eta_ap_ON"), float):
            self.telescopeDict["eta_ap_ON"] *= np.ones(nf_src)
        
        if isinstance(self.telescopeDict.get("eta_ap_OFF"), float):
            self.telescopeDict["eta_ap_OFF"] *= np.ones(nf_src)

        if self.telescopeDict["s_rms"] is not None:
            self.telescopeDict["s_rms"] *= 1e-6 # Convert um to m

            eta_surf = np.exp(-(4 * np.pi * self.telescopeDict["s_rms"] * f_arr / self.c)**2)

            self.telescopeDict["eta_ap_ON"] *= eta_surf 
            self.telescopeDict["eta_ap_OFF"] *= eta_surf 
        
        self.telescopeDict["dAz_chop"] /= 3600
        self.telescopeDict["Ax"] /= 3600
        self.telescopeDict["Axmin"] /= 3600
        self.telescopeDict["Ay"] /= 3600
        self.telescopeDict["Aymin"] /= 3600

    def runSimulation(self, t_obs, verbosity=1, outpath="./out/", overwrite=False):
        """!
        Run a gateau simulation.

        This is the main routine of gateau and should be called after filling all dictionaries and running the self.initialise() method.
        
        @param t_obs Total observation time in seconds.
        @param verbosity Level of verbosity for simulation.
            0           : no extra output w.r.t. logger.
            1 (default) : show execution times of important routines.
            2           : show execution times of important routines and memory transactions.
        @param outpath Directory to store output in.
        @param overwrite Whether to overwrite existing output directories. 
            If False (default), a prompt will appear to either overwrite or specify new directory.
        """

        if not self.initialisedSetup:
            self.clog.error("initSetup() MUST be called before running a simulation!")
            raise InitialError
            sys.exit()
        
        #### INITIALISING OBSERVATION PARAMETERS ####
        # Calculate number of time evaluations
        # Note that, in order to simplify TLS noise calculations, we make nTimes even
        nTimes = math.ceil(t_obs * self.instrumentDict["f_sample"])

        if nTimes % 2 == 1:
            nTimes -= 1
        
        t_range = 1 / self.instrumentDict["f_sample"] * np.arange(nTimes)

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
                if choice == "y":
                    shutil.rmtree(outpath)
                else:
                    outpath = input("\033[93mSpecify new output path: > ")


        self.clog.info("\033[1;32m*** STARTING gateau SIMULATION ***")
        
        start = time.time()

        gbind.run_gateau(self.instrumentDict, self.telescopeDict, 
                    self.atmosphereDict, self.sourceDict, nTimes, outpath)
        
        end = time.time()        
        
        self.clog.info("\033[1;32m*** FINISHED gateau SIMULATION ***")

