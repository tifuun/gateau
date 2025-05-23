/*! \file InterpUtils.h
 * \brief Utilities for linear interpolation.
 **/

#include <iostream>
#include <cmath>

#include "Structs.h"

#ifndef __INTERPUTILS_H
#define __INTERPUTILS_H

/**
  Linearly interpolate bivariate function.
 
  @param x Point in x to interpolate on.
  @param y Point in y to interpolate on.
  @param x0 Start of x-coordinates.
  @param y0 Start of y-coordinates.
  @param size_x Size of array containing x-coordinates.
  @param size_y Size of array containing y-coordinates.
  @param dx Stepsize of x.
  @param dy Stepsize of y.
  @param vals Function values on grid spanning x and y.
  @param size_vals Size of array containing values.
  @param debug Run method in debug mode. Default is false and is passed to findIndexLow.
  
  @returns val_interp Interpolated value of function on x0 and y0.
 */
double interpValue(double x, double y, ArrSpec<double> arrx, ArrSpec<double> arry, double *vals, int offset = 0, bool debug=false);

#endif
