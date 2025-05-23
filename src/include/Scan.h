/*! \file Scan.h
 * \brief File containing scanning patterns and their implementations.
 **/

#include <cmath>
#include <cstdio>
#include "Structs.h"

#ifndef __SCAN_h
#define __SCAN_h

# define M_PI           3.14159265358979323846  /* pi */

struct AzEl {
    double Az;      /**< Azimuth angle on-sky, in degrees.*/
    double El;      /**< Elevation angle in degrees.*/
};

struct xy_atm {
    double xAz;     /** < X-coordinate on atmosphere at reference height corresponding to Az.*/
    double yEl;     /** < Y-coordinate on atmosphere at reference height corresponding to El.*/
};

void sgn(double val, int &out);

/**
  Calculate new Azimuth-Elevation co-ordinate, accoding to chop position.

  This function just "scans" a single point, so seems sort of pointless. 
  Still implemented for completeness.

  @param center Az-El co-ordinate of point to observe, w.r.t. source Az-El.
  @param out Container for storing output Az-El co-ordinate.
  @param chop Whether chopper is in A (false) or B (true).
  @param sep Angular throw between chop A and B, in degrees.
 */
void scanPoint(AzEl* center, AzEl* out, bool chop, double sep = 0.);

/**
  Calculate new Azimuth-Elevation co-ordinate, accoding to chop position.

  The scan pattern used is the Daisy scan pattern.

  @param center Az-El co-ordinate of point to observe, w.r.t. source Az-El.
  @param out Container for storing output Az-El co-ordinate.
  @param telescope Telescope struct.
  @param t Time of pointing in seconds.
  @param chop Whether chopper is in A (false) or B (true).
  @param sep Angular throw between chop A and B, in degrees.
 */
void scanDaisy(AzEl* center, AzEl* out, Telescope<double> *telescope, double t, bool chop, double sep = 0.);

/**
  Calculate new Azimuth-Elevation co-ordinate, accoding to chop position.

  This function just "scans" a single point, so seems sort of pointless. 
  Still implemented for completeness.

  @param center Az-El co-ordinate of point to observe, w.r.t. source Az-El.
  @param out Container for storing output Az-El co-ordinate.
  @param chop Whether chopper is in A (false) or B (true).
  @param sep Angular throw between chop A and B, in degrees.
 */
void convertAnglesToSpatialAtm(AzEl* angles, xy_atm* out, double h_column);

void getABBA_posflag(double &t, AzEl *center, AzEl *pointing, Telescope<double> *telescope, int &flagout); 

void getONOFF_posflag(double &t, AzEl *center, AzEl *pointing, Telescope<double> *telescope, int &flagout);

void getnochop_posflag(double &t, AzEl *center, AzEl *pointing, Telescope<double> *telescope, int &flagout); 
#endif
