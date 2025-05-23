/*! \file CuScan.h
 * \brief File containing scanning patterns and their implementations for CUDA.
 **/

#include <math.h>
#include <cuda.h>

#ifndef __CUSCAN_h
#define __CUSCAN_h

/**
  Structure for storing an Azimuth-Elevation co-ordinate.
 */
struct AzEl {
    float Az;      /**< Azimuth angle on-sky, in degrees.*/
    float El;      /**< Elevation angle in degrees.*/
};                                                                                             
                                                                                               
                                                                                            
struct xy_atm {
    float xAz;     /** < X-coordinate on atmosphere at reference height corresponding to Az.*/
    float yEl;     /** < Y-coordinate on atmosphere at reference height corresponding to El.*/
};

#endif
