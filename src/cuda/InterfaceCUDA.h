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

#include "CuInterpUtils.h"
#include "FileIO.h"
#include "Structs.h"

#ifdef _WIN32
#   define GATEAU_DLL __declspec(dllexport)
#else
#   define GATEAU_DLL
#endif

#ifndef __InterfaceCUDA_h
#define __InterfaceCUDA_h

extern "C"
{
    GATEAU_DLL void run_gateau(Instrument *instrument, 
                               Telescope *telescope, 
                               Atmosphere *atmosphere, 
                               Source *source,
                               Cascade *cascade,
                               int nTimesTotal, 
                               char *outpath,
                               long long int seed);
}

#endif
