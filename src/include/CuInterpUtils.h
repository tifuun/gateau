/*! \file CuInterpUtils.h
 * \brief Utilities for linear interpolation for CUDA.
 **/

#include "cuda.h"
#include "math.h"
#include "Structs.h"

#ifndef __CUINTERPUTILS_H
#define __CUINTERPUTILS_H

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
  
  @returns val_interp Interpolated value of function on x0 and y0.
 */
__device__ void interpValue(float x, float y, 
            ArrSpec *arrx, ArrSpec *arry,
            float *vals, int offset, float &out) {
    
    int idx_x = floorf((x - arrx->start) / arrx->step);
    int idx_y = floorf((y - arry->start) / arry->step);
    
    float t = (x - (arrx->start + arrx->step*idx_x)) / arrx->step;
    float u = (y - (arry->start + arry->step*idx_y)) / arry->step;

    out =  (1-t)*(1-u) * vals[idx_x * arry->num + idx_y + offset];
    out += t*(1-u) * vals[(idx_x + 1) * arry->num + idx_y + offset];
    out += (1-t)*u * vals[idx_x * arry->num + idx_y + 1 + offset];
    out += t*u * vals[(idx_x + 1) * arry->num + idx_y + 1 + offset];
}

#endif
