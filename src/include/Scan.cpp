/*! \file Scan.cpp
 * \brief Definition of scanning patterns and their implementations.
 **/

#include "Scan.h"

void sgn(double val, int &out) {
    out = (double(0) < val) - (val < double(0));
}

void scanPoint(AzEl* center, AzEl* out, bool chop, double sep) {
    double offset = 0.;
    
    if (chop) {
        offset = sep;
    }    
    
    out->Az = center->Az + offset;
    out->El = center->El;
}

void scanDaisy(AzEl* center, AzEl* out, Telescope<double> *telescope, double t, bool chop, double sep) {
    double offset = 0.;
    
    if (chop) {
        offset = sep;
    }    
    
    out->Az = center->Az + offset 
        + telescope->Ax*sin(telescope->wx*t)*cos(telescope->wx*t + telescope->phix) 
        + telescope->Axmin*sin(telescope->wxmin*t)*cos(telescope->wxmin*t + telescope->phix);
    out->El = center->El 
        + telescope->Ay*sin(telescope->wy*t)*sin(telescope->wy*t + telescope->phiy) 
        + telescope->Aymin*sin(telescope->wymin*t)*sin(telescope->wymin*t + telescope->phiy) 
        - telescope->Ay;
}

void convertAnglesToSpatialAtm(AzEl* angles, xy_atm* out, double h_column) {
    
    double coord = tan(M_PI * angles->Az / 180.) * h_column;
    
    out->xAz = coord;
    coord = tan(M_PI * angles->El / 180.) * h_column;
    out->yEl = coord;
}

void getABBA_posflag(double &t, AzEl *center, AzEl *pointing, Telescope<double> *telescope, int &flagout) { 
    int n_chop;
    int n_nod;
    int position;

    bool chop_flag;

    double is_in_lower_half;
    int nod_flag;

    n_chop = floor(t * telescope->freq_chop);
    n_nod = floor(t * telescope->freq_nod);
    
    chop_flag = (n_chop % 2 != 0); // If even (false), ON. Odd (true), OFF.
    nod_flag = -1 + 2 * (n_nod % 2 != 0); // If even (false), AB. Odd (true), BA.
    //printf("%d \n", chop_flag);
    
    is_in_lower_half = (t - n_nod / telescope->freq_nod) - (1 / telescope->freq_nod / 2);
    sgn(is_in_lower_half, position);
    position *= nod_flag;
    
    scanPoint(center, pointing, chop_flag, position * telescope->dAz_chop);
    flagout = chop_flag * position + (1 - chop_flag) * (1 - position);
}

void getONOFF_posflag(double &t, AzEl *center, AzEl *pointing, Telescope<double> *telescope, int &flagout) {
    int n_chop;
    bool chop_flag;

    n_chop = floor(t * telescope->freq_chop);
    
    chop_flag = (n_chop % 2 != 0); // If even (false), ON. Odd (true), OFF.
    if(telescope->scantype == 0) {scanPoint(center, pointing, chop_flag, telescope->dAz_chop);}
    else if(telescope->scantype == 1) {scanDaisy(center, pointing, telescope, t, chop_flag, telescope->dAz_chop);}
    flagout = chop_flag;
}

void getnochop_posflag(double &t, AzEl *center, AzEl *pointing, Telescope<double> *telescope, int &flagout) { 
    if(telescope->scantype == 0) {scanPoint(center, pointing, 0, telescope->dAz_chop);}
    else if(telescope->scantype == 1) {scanDaisy(center, pointing, telescope, t, 0, telescope->dAz_chop);}
    flagout = 0;
}



