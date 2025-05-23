/*! \file InterfaceCUDA.h
    \brief Declarations of TiEMPO2 library for GPU.

    Provides single precision interface for NVIDIA GPUs running CUDA. 
*/
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <string>

#include "cuda.h"
#include "curand_kernel.h"

#include "CuScan.h"
#include "CuInterpUtils.h"
#include "Timer.h"
#include "FileIO.h"
#include "Structs.h"
#include "Filterbank.h"

#define CEFFSSIZE 4

#ifdef _WIN32
#   define TIEMPO2_DLL __declspec(dllexport)
#else
#   define TIEMPO2_DLL
#endif

#ifndef __InterfaceCUDA_h
#define __InterfaceCUDA_h

extern "C"
{
    TIEMPO2_DLL void runTiEMPO2_CUDA(Instrument<float> *instrument, 
                                     Telescope<float> *telescope, 
                                     Atmosphere<float> *atmosphere, 
                                     Source<float> *source,
                                     Cascade<float> *cascade,
                                     int nTimesTotal, 
                                     char *outpath);
}

#endif
