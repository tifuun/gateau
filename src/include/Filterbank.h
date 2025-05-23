#include "Structs.h"

#ifndef __Filterbank_h
#define __Filterbank_h

template<typename T>
void filterbankLorentz(Instrument<T> *instrument);

#endif

template<typename T>
void filterbankLorentz(Instrument<T> *instrument, T *output) {
    T gamma;
    T f_ch;
    T eta_ch;
    T df;
    T res;
    T res_out;

    for(int i=0; i<instrument->nf_ch; i++) {
        f_ch = instrument->f_ch[i];
        gamma = f_ch / (2 * instrument->R); 
        eta_ch = instrument->eta_filt[i];
        for(int j=0; j<instrument->f_spec.num; j++) {
            df = instrument->f_spec.start + instrument->f_spec.step*j - f_ch;
            res = eta_ch * (gamma*gamma / (df*df + gamma*gamma));
            res_out = res;
            for(int n=0; n<instrument->order-1; n++) {
                res_out *= res;
            }
            output[i * instrument->f_spec.num + j] = res_out;
        }
    }
}


