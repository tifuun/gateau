/*! \file InterfaceCPU.h
    \brief Declarations of library functions for simulations on CPU.
*/

#include <thread>
#include <vector>
#include <random>

#include "InterpUtils.h"
#include "Structs.h"
#include "Scan.h"
#include "Timer.h"
#include "FileIO.h"
#include "Filterbank.h"

#ifdef _WIN32
#   define TIEMPO2_DLL __declspec(dllexport)
#else
#   define TIEMPO2_DLL
#endif

#ifndef __InterfaceCPU_h
#define __InterfaceCPU_h

#define SI_TO_MJY               1E20 /* SI to MJy*/

#define PI 3.14159265358979323846  /* pi */
#define CL 2.9979246E8 // m s^-1
#define HP 6.62607015E-34
#define KB 1.380649E-23

extern "C"
{
    TIEMPO2_DLL void calcW2K(Instrument<double> *instrument, Telescope<double> *telescope, 
                             Atmosphere<double> *atmosphere, CalOutput<double> *output,
                             int nPWV, int nThreads);
       
    TIEMPO2_DLL void getSourceSignal(Instrument<double> *instrument, Telescope<double> *telescope, 
                                     double *output, double *I_nu, 
                                     double PWV, bool ON);
    
    TIEMPO2_DLL void getEtaAtm(ArrSpec<double> f_src, double *output, double PWV);

    TIEMPO2_DLL void getNEP(Instrument<double> *instrument, Telescope<double> *telescope, 
                            double *output, double PWV, double Tatm);

    TIEMPO2_DLL void getChopperCalibration(Instrument<double> *instrument, double *output, double Tchopper);

}

void parallelJobsW2K(Instrument<double> *instrument, Atmosphere<double> *atmosphere, 
                     CalOutput<double> *output, double *eta_atm, 
                     ArrSpec<double> PWV_atm, ArrSpec<double> f_atm, Effs<double> *effs, 
                     int nPWV, int start, int stop, double dPWV,
                     double* I_atm, double* I_gnd, double* I_tel, double *I_CMB, int threadIdx);

#endif
