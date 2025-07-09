/*! \file CuInterpUtils.h
 * \brief Utilities for linear interpolation for CUDA.
 **/
#include <stdexcept>

#include "cuda.h"
#include "math.h"
#include "structs.h"

#include <gsl/gsl_fit.h>

#define NPWV            1000
#define PWV_START       0.1
#define PWV_END         5.5
#define DPWV            0.0054

#ifndef __UTILITIES_H
#define __UTILITIES_H

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
__host__ __device__ void interpValue(float x, float y, 
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

/**
  Cascade a PSD through a reflector system, and couple to a specific parasitic PSD.

  @param P_nu_in PSD of incoming signal to be cascaded.
  @param eta Efficiency term associated with cascade.
  @param T_parasitic Temperature of parasitic source.
  @param nu Frequency in Hz.

  @returns Cascade output PSD.
 */
__host__ __device__ float rad_trans(float psd_in, 
                                      float eta, 
                                      float psd_parasitic)
{
    return eta * psd_in + (1 - eta) * psd_parasitic;
}

__host__ void resp_calibration(int start, 
                    int stop,
                    ArrSpec *f_atm, 
                    ArrSpec *pwv_atm, 
                    ArrSpec *f_src,
                    float Tp_atm,
                    int nf_ch,
                    float *eta_cascade,
                    float *psd_cascade,
                    int num_stage,
                    float *psd_atm,
                    float *eta_atm,
                    float *filterbank,
                    float *eta_kj_sum,
                    float *Psky, 
                    float *Tsky)
{
    // FLOATS
    float eta_atm_interp;   // Interpolated eta_atm, over frequency and PWV
    float freq;             // Bin frequency
    float psd_in;           // Local variable for storing PSD.
    float psd_in_k;         // Local variable for calculating psd per channel
    float eta_kj;           // Filter efficiency for bin j, at channel k.
    float psd_parasitic_use;
    float pwv_loc;
    float psd_atm_loc;
    float temp1, temp2;

    for(int idx=start; idx<stop; idx++)
    {
        pwv_loc = PWV_START + idx*DPWV;
        for(int idy=0; idy<f_src->num; idy++)
        {
            freq = f_src->start + f_src->step * idy;

            interpValue(pwv_loc, freq,
                        pwv_atm, f_atm,
                        eta_atm, 0, eta_atm_interp);
            
            psd_atm_loc = psd_atm[idy];

            // Initial pass through atmosphere
            psd_in = rad_trans(0., eta_atm_interp, psd_atm_loc);

            // Radiative transfer cascade
            for (int n=0; n<num_stage; n++) 
            {
                psd_parasitic_use = psd_cascade[n*f_src->num + idy];
                if (psd_parasitic_use < 0) 
                {
                    psd_parasitic_use = eta_atm_interp * psd_atm_loc;
                }

                psd_in = rad_trans(psd_in, eta_cascade[n*f_src->num + idy], psd_parasitic_use);
            }

            temp1 = eta_cascade[num_stage*f_src->num + idy];
            temp2 = psd_cascade[num_stage*f_src->num + idy];

            for(int k=0; k<nf_ch; k++) {
                eta_kj = filterbank[k*f_src->num + idy];

                psd_in_k = rad_trans(psd_in, eta_kj*temp1, temp2);

                Psky[k*NPWV + idx] += psd_in_k * f_src->step; 
                Tsky[k*NPWV + idx] += eta_kj * (1 - eta_atm_interp) * Tp_atm / eta_kj_sum[k]; 
            }
        }
    }
}

__host__ void fit_calibration(float *Psky,
        float *Tsky,
        int nf_ch,
        float *c0,
        float *c1)
{
    double c0_loc, c1_loc, cov00, cov01, cov11, sumsq;

    double *Psky_k = new double[NPWV]; 
    double *Tsky_k = new double[NPWV]; 

    for(int k=0; k<nf_ch; k++)
    {
        for(int j=0; j<NPWV; j++)
        {
            Psky_k[j] = static_cast<double>(Psky[k*NPWV + j]);
            Tsky_k[j] = static_cast<double>(Tsky[k*NPWV + j]);
        }
        gsl_fit_linear(Psky_k, 1, Tsky_k, 1, NPWV, &c0_loc, &c1_loc, &cov00, &cov01, &cov11, &sumsq);
        c0[k] = static_cast<float>(c0_loc);
        c1[k] = static_cast<float>(c1_loc);
    }
    delete[] Psky_k;   
    delete[] Tsky_k;   
}

#endif
