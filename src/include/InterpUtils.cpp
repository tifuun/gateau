/*! \file InterpUtils.cpp
 * \brief Utilities for linear interpolation.
 **/

#include "InterpUtils.h"

double interpValue(double x, double y, ArrSpec<double> arrx, ArrSpec<double> arry, double *vals, int offset, bool debug) {
    
    int idx_x = floorf((x - arrx.start) / arrx.step);
    int idx_y = floorf((y - arry.start) / arry.step);

    double f00 = vals[idx_x * arry.num + idx_y + offset];
    double f10 = vals[(idx_x + 1) * arry.num + idx_y + offset];
    double f01 = vals[idx_x * arry.num + idx_y + 1 + offset];
    double f11 = vals[(idx_x + 1) * arry.num + idx_y + 1 + offset];
    
    double t = (x - (arrx.start + arrx.step*idx_x)) / arrx.step;
    double u = (y - (arry.start + arry.step*idx_y)) / arry.step;
    
    double fxy = (1-t)*(1-u)*f00 + t*(1-u)*f10 + t*u*f11 + (1-t)*u*f01;
    //printf("%.12e %.12e %.12e\n", x, y, fxy);

    return fxy;
}
