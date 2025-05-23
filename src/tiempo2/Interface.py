import math
import os
import shutil
import time
import copy 
import sys
import numpy as np
import matplotlib.pyplot as pt
import scipy.fft as fft
import scipy.signal as signal
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d

import tiempo2.Filterbank as TFilter
import tiempo2.InputChecker as TCheck
import tiempo2.Atmosphere as TAtm
import tiempo2.BindCPU as TBCPU
import tiempo2.RemoveATM as TRemove
import tiempo2.Parallel as TParallel
import tiempo2.Utils as TUtils

import psutil
import logging
from tiempo2.CustomLogger import CustomLogger

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
    Interface for TiEMPO2 simulations

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
    w2k_set             = False
    
    initialisedSetup    = False
    initialisedObserve  = False

    clog_mgr = CustomLogger(os.path.basename(__file__))
    clog = clog_mgr.getCustomLogger()

    outPath = os.getcwd() # Output path defaults to cwd    
    
    c = 2.99792458e8

    def __init__(self, verbose=True, outPath=None):
        """!
        Create an interface object for TiEMPO2.

        @param verbose Set verbosity of logger. If False, will not log any output to screen.
            If True, will log simulation information, warnings, and errors to screen. 
            Default is True.
        @param outPath Path to where the interface object will write output. 
            Default is the current working directory.
        """

        self.outPath = outPath if outPath is not None else self.outPath
        if not verbose:
            self.clog.setLevel(logging.CRITICAL)
    

    def _conv(self, arg):
        return arg
    
    def setLoggingVerbosity(self, verbose=True):
        """!
        Set or change the verbosity of the TiEMPO2 logger.
        
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

        errlist = TCheck.checkTelescopeDict(telescopeDict)

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

        errlist = TCheck.checkInstrumentDict(instrumentDict)

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

        errlist = TCheck.checkAtmosphereDict(atmosphereDict)

        if not errlist:
            self.__atmosphereDict = atmosphereDict
            self.atmosphere_set = True

        else:
            errstr = f"Errors encountered in atmosphere dictionary in fields :{errlist}."
            raise FieldError(errstr)

    def prepareAtmosphere(self):
        """!
        Prepare the ARIS screen chunks for usage in TiEMPO2 simulations.
        A telescope and atmosphere dictionary need to be set, otherwise an error is raised.
        
        This function loads the ARIS screens, convolves them with an apodized Gaussian, the size of the telescope aperture and an edge taper of -10 dB (this will be adjustable in future updates). 
        Then, the prepared ARIS screens are saved in the same directory, under the 'prepd' subdirectory. 
        The original names are stripped, and only the screen index is used as name. The extension type is changed from .dat into .datp.

        For each simulation setup (disregarding source) and ARIS screen set, this function only needs to be run once. 
        For subsequent simulations the prepared screens can be reused.
        
        @ingroup initialise
        """

        if not (self.telescope_set and
                self.atmosphere_set):
            
            self.clog.error("a telescope and atmosphere dictionary MUST be set before calling prepareAtmosphere()")
            raise InitialError
            sys.exit()

        TAtm.prepAtmospherePWV(self.__atmosphereDict, self.__telescopeDict, self.clog)

    # NUMBER PARAMETER IS TEMPORARY
    def initSetup(self, use_ARIS=True, number=1, start=0):
        """!
        Initialise a TiEMPO2 setup. 

        The idea is that a TiEMPO2 setup can be done sequentially, the first step being a setup of the terrestrial part.
        This means that the first setup should consist of everything on Earth: instrument, telescope, and atmosphere.
        Therefore, this is the first initialisation that should be performed.

        This function will check wether or not the instrument, telescope, and atmosphere are set.
        If not, an InitialError will be raised.

        @param use_ARIS Whether to load an ARIS screen or not. 
            Some functions of TiEMPO2 do not require an ARIS screen to be loaded. 
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
        Initialise a TiEMPO2 source. 

        The idea is that a TiEMPO2 setup can be done sequentially, the first step being a setup of the terrestrial part.
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
                
        self.instrumentDict["filterbank"] = TFilter.generateFilterbankFromR(self.instrumentDict)
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

    def getSourceSignal(self, Az_point, El_point, PWV_value=-1, ON=True, w2k=False):
        """!
        Get astronomical signal without atmospheric noise, but with all efficiencies, atmospheric transmission and filterbank.

        @param Az_point Azimuth point on-sky in arcsec where source should be evaluated.
        @param El_point Elevation point on-sky in arcsec where source should be evaluated.
        @param PWV_value PWV value for atmospheric transmission. Defaults to -1 (no atmosphere).
        @param ON Whether to evaluate source in ON path (default), or OFF. Makes a difference when different eta_ap have been defined for each path.

        @returns Transmitted signal (SI) and its frequency range (Hz).

        @ingroup auxilliarymethods
        """
        
        #SZ, Az, El = TSource.generateSZMaps(self.sourceDict, self.instrumentDict, self.clog, telescopeDict=self.telescopeDict, trace_src=trace_src)
        #SZ = np.squeeze(SZ) 

        SZ = self.sourceDict["I_nu"][15,15,:] - self.sourceDict["I_nu"][0,0,:] 

        res = TBCPU.getSourceSignal(self.instrumentDict, self.telescopeDict, self.atmosphereDict, SZ, PWV_value, ON)
        
        if w2k: 
            if not self.w2k_set:
                self.clog.error("in order to convert Watts to Kelvin, a call to calcW2K has to be made first!")
                raise InitialError
                sys.exit()

            res = self.Watt2Kelvin(res)

        return res, self.instrumentDict["f_ch_arr"]
    
    def getChopperCalibration(self, Tcal):
        """!
        Get power emitted by a blackbody source in front of cryostat.
        For calibration purposes.
 
        @ingroup auxilliarymethods       
        """

        res = TBCPU.getChopperCalibration(self.instrumentDict, Tcal)
        return res
    
    def getEtaAtm(self, PWV_value):
        """!
        Get atmosphere transmission at source frequencies given a PWV.

        @param PWV_value PWV value for atmospheric transmission. 

        @returns Atmospheric transmission and its frequency range (Hz).
 
        @ingroup auxilliarymethods       
        """

        res = TBCPU.getEtaAtm(self.instrumentDict, PWV_value)
        return res, self.instrumentDict["f_src"]

    def getNEP(self, PWV_value):
        """!
        Calculate Noise Equivalent Power (NEP) from the atmosphere.

        @param PWV_value PWV value at which to calculate atmospheric NEP.

        @returns NEP (SI) and its frequency range (Hz).
 
        @ingroup auxilliarymethods       
        """

        res = TBCPU.getNEP(self.instrumentDict, self.telescopeDict, self.atmosphereDict, PWV_value)
        return res, self.instrumentDict["f_ch_arr"]

    def runSimulation(self, t_obs, device="CPU", nThreads=None, verbosity=1, outpath="./out/", overwrite=False):
        """!
        Run a TiEMPO2 simulation.

        This is the main routine of TiEMPO2 and should be called after filling all dictionaries and running the self.initialise() method.
        
        @param t_obs Total observation time in seconds.
        @param device Whether to run on CPU (device='CPU') or GPU (device='GPU'). Default is 'CPU'.
        @param nThreads Number of threads to use. Default is 1. Only relevant when device='CPU'.
        @param verbosity Level of verbosity for simulation.
            0           : no extra output w.r.t. logger.
            1 (default) : show execution times of important routines.
            2           : show execution times of important routines and memory transactions.
        @param outpath Directory to store output in.
        @param overwrite Whether to overwrite existing output directories. 
            If False (default), a prompt will appear to either overwrite or specify new directory.
        """

        nThreads = 1 if nThreads is None else nThreads

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


        self.clog.info("\033[1;32m*** STARTING TiEMPO2 SIMULATION ***")
        
        start = time.time()

        if device == "CPU":
            res = TBCPU.runTiEMPO2(self.instrumentDict, self.telescopeDict, 
                        self.atmosphereDict, self.sourceDict, nTimes, nThreads)
        
        elif device == "GPU":
            TBCPU.runTiEMPO2_CUDA(self.instrumentDict, self.telescopeDict, 
                        self.atmosphereDict, self.sourceDict, nTimes, outpath)
        
        end = time.time()        
        
        self.clog.info("\033[1;32m*** FINISHED TiEMPO2 SIMULATION ***")

    def calcW2K(self, nPWV, nThreads=None, verbosity=1):
        """!
        Calculate a Watt-to-Kelvin calibration table..

        @param nPWV Number of PWV points to consider.
        @param nThreads Number of threads to use. Default is 1.
        @param verbosity Level of verbosity for simulation.
            0           : no extra output w.r.t. logger.
            1 (default) : show execution times of important routines.
            2           : show execution times of important routines and memory transactions.

        @ingroup auxilliarymethods
        """

        nThreads = 1 if nThreads is None else nThreads
        
        if not self.initialisedSetup:
            self.clog.error("initSetup() MUST be called before running a W2K calibration!")
            raise InitialError
            sys.exit()

        self.clog.info("\033[1;32m*** STARTING TiEMPO2 W2K CALIBRATION ***")
        
        start = time.time()

        res = TBCPU.calcW2K(self.instrumentDict, self.telescopeDict, 
                            self.atmosphereDict, nPWV, nThreads)
        
        a = np.zeros(self.instrumentDict["nf_ch"])
        b = np.zeros(self.instrumentDict["nf_ch"])
        for k in range(self.instrumentDict["nf_ch"]):
            try:
                _a, _b = np.polyfit(res["power"][:,k], res["temperature"][:,k], 1)
            except:
                _a = 0
                _b = 0
            a[k] = _a
            b[k] = _b

        res["a"] = a
        res["b"] = b

        end = time.time()        

        self.w2k_conv = res
        self.w2k_set = True

        self.clog.info("\033[1;32m*** FINISHED TiEMPO2 W2K CALIBRATION ***")
    
    def Watt2Kelvin(self, output):
        """!
        Convert the signal in output from a Watt to a Kelvin temperature scale.

        This function updates the content of the 'signal' field of the output dictionary, 
        so if you want to keep the signal in Watts as well, make sure to (deep)copy it to a different array first.

        @param output An output dictionary containing the signal.
        @param w2k A Watt-to-Kelvin (w2k) dictionary.

        @ingroup auxilliarymethods
        """
        
        try:
            output["signal"] = self.w2k_conv["a"] * output["signal"] + self.w2k_conv["b"]

        except:
            output = self.w2k_conv["a"] * output + self.w2k_conv["b"]
    
    ####  UTILITY FUNCTIONS

    def shuffle_w2k(self, n_shuffle, n_avg):
        if not self.w2k_set:
            self.clog.error("in order to shuffle a w2k calibration, a call to calcW2K has to be made first!")
            raise InitialError
            sys.exit()

        n_chan = self.w2k_conv["a"].size

        if n_shuffle > n_chan:
            self.clog.warning("Attempting to shuffle more channels than are available, reducing amount of channels to shuffle.")
            n_shuffle = n_chan

        to_shuffle = np.random.choice(np.arange(n_chan), n_shuffle, replace=False)
        
        n_same = n_avg - n_shuffle

        a_avg = (np.nansum(self.w2k_conv["a"][to_shuffle]) + n_same*self.w2k_conv["a"][to_shuffle]) / n_avg
        b_avg = (np.nansum(self.w2k_conv["b"][to_shuffle]) + n_same*self.w2k_conv["b"][to_shuffle]) / n_avg

        self.w2k_conv["a"][to_shuffle] = a_avg
        self.w2k_conv["b"][to_shuffle] = b_avg

        return to_shuffle

    def calcSignalPSD(self, outpath, num_threads=1, nperseg=2**11, w2k=False):
        """!
        Calculate signal PSD of a simulation output.

        @param output Output structure obtained from a callc to runSimulation.
        @param x Array over which output signal is defined.
        @param axis Axis over which to calculate signal PSD. 0 is time axis (constant channel), 1 is frequency axis (constant timeslice).

        @returns signal_psd Signal PSD.
        @returns freq_psd Frequencies at which the signal PSD is defined, in Hertz.
 
        @ingroup auxilliarymethods       
        """

        if not self.instrument_set:
            
            self.clog.error("the instrument with which the data was simulated must be set in the interface!")
            raise InitialError
            sys.exit()

        conv = self._conv
        
        if w2k: 
            if not self.w2k_set:
                self.clog.error("in order to convert Watts to Kelvin, a call to calcW2K has to be made first!")
                raise InitialError
                sys.exit()

            conv = self.Watt2Kelvin
        
        self.clog.info("Calculating signal PSD.")
        
        Pxx, f_Pxx = TParallel.parallel_job(outpath, 
                                      num_threads = num_threads, 
                                      job = TUtils.calcSignalPSD, 
                                      conv = conv,
                                      args_list = [self.instrumentDict["f_sample"], nperseg])
        
        arr_sizes = np.array([x.size for x in f_Pxx])

        tot_Pxx = np.nanmean(Pxx[arr_sizes == np.max(arr_sizes)], axis=0)
        tot_f_Pxx = np.nanmean(f_Pxx, axis=0)

        return tot_Pxx, tot_f_Pxx
    
    def convolveSourceCube(self, source_cube, az_grid, el_grid, f_arr, Dtel, num_threads=1):
        """!
 
        @ingroup auxilliarymethods       
        """
        
        self.clog.info("Convolving source cube.")
        
        source_cube_convolved = TParallel.parallel_job_np(source_cube, 
                                      num_threads = num_threads, 
                                      job = TUtils.convolveSourceCube, 
                                      arr_par = f_arr,
                                      args_list = [Dtel, az_grid, el_grid],
                                      axis = -1)

        return source_cube_convolved

    def rebinSignal(self, output, freqs_old, nbins_add, final_bin=True):
        """!
        Rebin a simulation result into a coarser bin size.
        
        @param final_bin If number of old bins divided by nbins_add is not an integer, wether to rebin final new bin with less bins, or add extra bins to second-to-last bin.
        """

        shape_old  = output.get("signal").shape
        nbins_old = shape_old[1]
        
        if (not isinstance(nbins_add, int)) or (nbins_add == 1):
            self.clog.error(f"Rebin number must be an integer and larger than 1.")
            exit(1)

        if final_bin:
            nbins_new = math.ceil(nbins_old / nbins_add)
        else:
            nbins_new = math.floor(nbins_old / nbins_add)

        signal_new = np.zeros((shape_old[0], nbins_new))
        freqs_new = np.zeros(nbins_new)

        for nbin in range(nbins_new):
            start_bin = nbin * nbins_add
            if nbin == nbins_new - 1:
                signal_new[:,nbin] = np.mean(output.get("signal")[:,start_bin:], axis=1)
                freqs_new[nbin] = np.mean(freqs_old[start_bin:])

            else:
                signal_new[:,nbin] = np.mean(output.get("signal")[:,start_bin:start_bin+nbins_add], axis=1)
                freqs_new[nbin] = np.mean(freqs_old[start_bin:start_bin+nbins_add])

        output_binned = copy.deepcopy(output)
        output_binned["signal"] = signal_new

        return output_binned, freqs_new
    
    def avgTOD(self, outpath, num_threads=1, w2k=False):
        """!
        Apply full time-averaging and direct atmospheric subtraction.

        @param outpath Path to where output from a simulation is saved.
        @param resolution How far to average and subtract. The following options are available:
            0: Reduce signal to a single spectrum, fully averaged and subtracted according to the chopping/nodding scheme.
            1: Reduce signal by averaging over ON-OFF chop positions.
        @param num_threads Number of threads to use. If this number is more than necessary, it will automatically scale down.

        @returns red_signal Reduced signal.
        @returns red_Az Reduced Azimuth array. Only returned if resolution == 1.
        @returns red_El Reduced Elevation array. Only returned if resolution == 1.
        """
        
        if not self.instrument_set:
            self.clog.error("the instrument with which the data was simulated must be set in the interface!")
            raise InitialError
            sys.exit()
        
        conv = self._conv
        
        if w2k: 
            if not self.w2k_set:
                self.clog.error("in order to convert Watts to Kelvin, a call to calcW2K has to be made first!")
                raise InitialError
                sys.exit()

            conv = self.Watt2Kelvin
        
        self.clog.info("Applying time-averaging.")
        
        avg_l, var_l, N_l = TParallel.parallel_job(outpath, 
                                      num_threads = num_threads, 
                                      job = TUtils.avgTOD, 
                                      conv = conv,
                                      args_list = [self.instrumentDict["f_sample"]])
        
        avg = np.nansum((N_l - 1) * avg_l.T, axis=1) / (np.nansum(N_l) - len(N_l))
        var = np.nansum((N_l - 1) * var_l.T, axis=1) / (np.nansum(N_l) - len(N_l))**2

        return avg, var, self.instrumentDict["f_ch_arr"]
    
    def avgDirectSubtract(self, outpath, resolution=0, num_threads=1, var_method="PSD", w2k=False):
        """!
        Apply full time-averaging and direct atmospheric subtraction.

        @param outpath Path to where output from a simulation is saved.
        @param resolution How far to average and subtract. The following options are available:
            0: Reduce signal to a single spectrum, fully averaged and subtracted according to the chopping/nodding scheme.
            1: Reduce signal by averaging over ON-OFF chop positions.
        @param num_threads Number of threads to use. If this number is more than necessary, it will automatically scale down.

        @returns red_signal Reduced signal.
        @returns red_Az Reduced Azimuth array. Only returned if resolution == 1.
        @returns red_El Reduced Elevation array. Only returned if resolution == 1.
        """
        
        if not self.instrument_set:
            self.clog.error("the instrument with which the data was simulated must be set in the interface!")
            raise InitialError
            sys.exit()
        
        conv = self._conv
        
        if w2k: 
            if not self.w2k_set:
                self.clog.error("in order to convert Watts to Kelvin, a call to calcW2K has to be made first!")
                raise InitialError
                sys.exit()

            conv = self.Watt2Kelvin
        
        if resolution == 0:
            self.clog.info("Applying time-averaging and direct subtraction.")
            
            avg_l, var_l, N_l = TParallel.parallel_job(outpath, 
                                          num_threads = num_threads, 
                                          job = TUtils.avgDirectSubtract, 
                                          conv = conv,
                                          args_list = [self.instrumentDict["f_sample"], var_method])
            
            avg = np.nansum((N_l - 1) * avg_l.T, axis=1) / (np.nansum(N_l) - len(N_l))
            var = np.nansum((N_l - 1) * var_l.T, axis=1) / (np.nansum(N_l) - len(N_l))**2

            return avg, var, self.instrumentDict["f_ch_arr"]

        elif resolution == 1:
            self.clog.info("Averaging and subtracting over ON-OFF pairs.")
            red_signal, red_Az, red_El = TRemove.avgDirectSubtract_chop(output)
            
            return red_signal, red_Az, red_El

    def getExposureTime(self, red_Az, red_El, nAz_grid, nEl_grid):
        points = np.array([red_Az, red_El])

        min_Az = np.min(red_Az)
        max_Az = np.max(red_Az)
        min_El = np.min(red_El)
        max_El = np.max(red_El)

        grid_Az, grid_El = np.mgrid[min_Az:max_Az:nAz_grid*1j, min_El:max_El:nEl_grid*1j]
        statObj = binned_statistic_2d(red_Az, red_El, np.ones(red_Az.size), bins=[nAz_grid, nEl_grid], statistic='count') 
        exposure_time = statObj.statistic / self.instrumentDict.get("f_sample")
    
        return exposure_time, grid_Az, grid_El

    def regridRedSignal(self, red_signal, red_Az, red_El, nAz_grid, nEl_grid, stack=False, idx=None):
        points = np.array([red_Az, red_El])

        min_Az = np.min(red_Az)
        max_Az = np.max(red_Az)
        min_El = np.min(red_El)
        max_El = np.max(red_El)

        grid_Az, grid_El = np.mgrid[min_Az:max_Az:nAz_grid*1j, min_El:max_El:nEl_grid*1j]

        if idx is None:
            if stack:
                grid_signal = binned_statistic_2d(red_Az, red_El, np.mean(red_signal, axis=1), bins=[nAz_grid, nEl_grid]).statistic 

            else:
                grid_signal = np.zeros((nAz_grid, nEl_grid, red_signal.shape[1]))
                for i in range(red_signal.shape[1]):
                    grid_signal[:,:,i] = binned_statistic_2d(red_Az, red_El, red_signal[:,i], bins=[nAz_grid, nEl_grid]).statistic 

        else:
            grid_signal = binned_statistic_2d(red_Az, red_El, red_signal[:,idx], bins=[nAz_grid, nEl_grid]).statistic 
        
        return grid_signal, grid_Az, grid_El












