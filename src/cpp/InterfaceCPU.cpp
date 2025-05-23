/*! \file InterfaceCPU.cpp
    \brief Implementations of library functions for simulations on CPU.

*/

#include "InterfaceCPU.h"

double inline getPlanck(double T, double nu)
{
    double prefac = 2 * HP * nu*nu*nu / (CL*CL);
    double dist = 1 / (exp(HP*nu / (KB*T)) - 1); 
    return prefac * dist;
}

TIEMPO2_DLL void calcW2K(Instrument<double> *instrument, Telescope<double> *telescope, 
            Atmosphere<double> *atmosphere, CalOutput<double> *output, int nPWV, int nThreads) {
    // ALLOCATIONS
    // Doubles 
    double freq;    // Frequency, used for initialising background sources.

    // Integers
    int step;       // Stepsize for each thread.
    int nf_src = instrument->f_spec.num;

    // Double array types
    double* I_atm = new double[nf_src];
    double* I_gnd = new double[nf_src];
    double* I_tel = new double[nf_src];
    double* I_CMB = new double[nf_src];

    // Initialise constant efficiency struct
    Effs<double> effs;
    effs.eta_tot_chain = instrument->eta_inst * instrument->eta_misc * telescope->eta_fwd * telescope->eta_mir * 0.5;
    effs.eta_tot_gnd = instrument->eta_inst  * instrument->eta_misc * (1 - telescope->eta_fwd) * telescope->eta_mir * 0.5;
    effs.eta_tot_mir = instrument->eta_inst  * instrument->eta_misc * (1 - telescope->eta_mir) * 0.5;

    // Make threadpool
    std::vector<std::thread> threadPool;
    threadPool.resize(nThreads);
    
    // PREAMBLE
    step = ceil(nPWV / nThreads);
    
    //printf("\033[1;32m\r");
    
    // Calculate I_atm, I_gnd, I_tel before entering time loop.
    // These stay constant during observation anyways.
    for(int j=0; j<nf_src; j++) { 
        freq = instrument->f_spec.start + instrument->f_spec.step * j;
        
        I_atm[j] = getPlanck(atmosphere->Tatm, freq); 
        I_gnd[j] = getPlanck(telescope->Tgnd, freq); 
        I_tel[j] = getPlanck(telescope->Ttel, freq);
        I_CMB[j] = getPlanck(2.725, freq);
    }
    
    // Allocate sub-arrays outside of thread loop - safer I guess
    Timer timer;

    timer.start();
    
    double *eta_atm;
    ArrSpec<double> PWV_atm;
    ArrSpec<double> f_atm;

    readEtaATM<double, ArrSpec<double>>(&eta_atm, &PWV_atm, &f_atm);

    double dPWV_arr = (PWV_atm.num * PWV_atm.step - PWV_atm.start ) / nPWV;

    // Main thread spawning loop
    for(int n=0; n < nThreads; n++) {
        int final_step; // Final step for 
        
        if(n == (nThreads - 1)) {
            final_step = nPWV;
        } else {
            final_step = (n+1) * step;
        }
        
        threadPool[n] = std::thread(&parallelJobsW2K, instrument, atmosphere, 
                output, eta_atm, PWV_atm, f_atm, &effs, nPWV, n * step, final_step, dPWV_arr,
                I_atm, I_gnd, I_tel, I_CMB, n);
    }

    // Wait with execution until all threads are done
    for (std::thread &t : threadPool) {
        if (t.joinable()) {
            t.join();
        }
    }

    timer.stop();
    //output->t_thread = timer.get();
    
    //printf("\033[0m\n");

    delete[] I_atm;
    delete[] I_gnd;
    delete[] I_tel;
    delete[] I_CMB;
    delete[] eta_atm;
}

TIEMPO2_DLL void getChopperCalibration(Instrument<double> *instrument, double *output, double Tcal) {
    double freq; // Bin frequency
    double eta_kj; // Filter efficiency for bin j, at channel k
    double I_cal;
    
    double PSD_nu;
    
    double eta_tot_chain = instrument->eta_inst * instrument->eta_misc * 0.5;
    
    for(int j=0; j<instrument->f_spec.num; j++) { 
        freq = instrument->f_spec.start + instrument->f_spec.step * j;
        I_cal = getPlanck(Tcal, freq); 

        PSD_nu = eta_tot_chain * I_cal * CL*CL / (freq*freq);
        for(int k=0; k<instrument->nf_ch; k++) {
            eta_kj = instrument->filterbank[k*instrument->f_spec.num + j];
            output[k] += PSD_nu * eta_kj * instrument->f_spec.step; 
        }
    }
}

TIEMPO2_DLL void getSourceSignal(Instrument<double> *instrument, Telescope<double> *telescope, 
            double *output, double *I_nu, double PWV, bool ON) {
    double freq; // Bin frequency
    double eta_kj; // Filter efficiency for bin j, at channel k
    double eta_atm_interp; // Interpolated eta_atm, over frequency and PWV
    
    double PSD_nu;
    
    double eta_tot_chain = instrument->eta_inst * instrument->eta_misc * telescope->eta_fwd * telescope->eta_mir * 0.5;
    double eta_ap;
    
    double *eta_atm;
    ArrSpec<double> PWV_atm;
    ArrSpec<double> f_atm;

    readEtaATM<double, ArrSpec<double>>(&eta_atm, &PWV_atm, &f_atm);
    
    for(int j=0; j<instrument->f_spec.num; j++) { 
        freq = instrument->f_spec.start + instrument->f_spec.step * j;
        
        eta_atm_interp = 1.;

        if(PWV > 0) { 
            eta_atm_interp = interpValue(PWV, freq, 
                    PWV_atm, f_atm, eta_atm);
        }
        
        if(ON) {
            eta_ap = telescope->eta_ap_ON[j];
        }

        else {
            eta_ap = telescope->eta_ap_OFF[j];
        }

        PSD_nu = eta_ap * eta_atm_interp * eta_tot_chain * I_nu[j] * CL*CL / (freq*freq);
        for(int k=0; k<instrument->nf_ch; k++) {
            eta_kj = instrument->filterbank[k*instrument->f_spec.num + j];
            output[k] += PSD_nu * eta_kj * instrument->f_spec.step; 
        }
    }
    delete[] eta_atm;
}

TIEMPO2_DLL void getEtaAtm(ArrSpec<double> f_src, double *output, double PWV) {
    double freq;

    double *eta_atm;
    ArrSpec<double> PWV_atm;
    ArrSpec<double> f_atm;


    readEtaATM<double, ArrSpec<double>>(&eta_atm, &PWV_atm, &f_atm);

    for(int j=0; j<f_src.num; j++)
    {   
        freq = f_src.start + f_src.step * j;
        output[j] = interpValue(PWV, freq, 
                PWV_atm, f_atm, eta_atm);
    }

    delete[] eta_atm;
}

TIEMPO2_DLL void getNEP(Instrument<double> *instrument, Telescope<double> *telescope, 
            double *output, double PWV, double Tatm) {
    // Double array types
    double I_atm;
    double I_gnd;
    double I_tel;

    // Initialise constant efficiency struct
    Effs<double> effs;
    effs.eta_tot_chain = instrument->eta_inst * instrument->eta_misc * telescope->eta_fwd * telescope->eta_mir * 0.5;
    effs.eta_tot_gnd = instrument->eta_inst  * instrument->eta_misc * (1 - telescope->eta_fwd) * telescope->eta_mir * 0.5;
    effs.eta_tot_mir = instrument->eta_inst  * instrument->eta_misc * (1 - telescope->eta_mir) * 0.5;
    
    double eta_atm_interp; // Interpolated eta_atm, over frequency and PWV
    double freq; // Bin frequency
    double eta_kj; // Filter efficiency for bin j, at channel k

    double PSD_back;
    
    double *eta_atm;
    ArrSpec<double> PWV_atm;
    ArrSpec<double> f_atm;

    readEtaATM<double, ArrSpec<double>>(&eta_atm, &PWV_atm, &f_atm);

    for(int j=0; j<instrument->f_spec.num; j++) { 
        freq = instrument->f_spec.start + instrument->f_spec.step * j;
        
        I_atm = getPlanck(Tatm, freq); 
        I_gnd = getPlanck(telescope->Tgnd, freq); 
        I_tel = getPlanck(telescope->Ttel, freq);
        
        eta_atm_interp = interpValue(PWV, freq, 
                PWV_atm, f_atm, eta_atm);
        
        PSD_back = (effs.eta_tot_chain * (1 - eta_atm_interp) * I_atm 
            + effs.eta_tot_gnd * I_gnd 
            + effs.eta_tot_mir * I_tel) * CL*CL / (freq*freq);
        
        for(int k=0; k<instrument->nf_ch; k++) {   
            eta_kj = instrument->filterbank[k*instrument->f_spec.num + j];
            
            output[k] += 2 * instrument->f_spec.step * PSD_back * eta_kj * (HP * freq + PSD_back * eta_kj + 2 * instrument->delta / instrument->eta_pb);
        }
    }
    
    for(int k=0; k<instrument->nf_ch; k++) {
        output[k] = sqrt(output[k]);
    }
    delete[] eta_atm;
}

void parallelJobsW2K(Instrument<double> *instrument, Atmosphere<double> *atmosphere, 
        CalOutput<double> *output, double *eta_atm, ArrSpec<double> PWV_atm, ArrSpec<double> f_atm,
        Effs<double> *effs, int nPWV, int start, int stop, double dPWV,
        double* I_atm, double* I_gnd, double* I_tel, double *I_CMB, int threadIdx) {
    
    // Get starting time and chop parameters
    double freq; // Bin frequency
    double eta_kj; // Filter efficiency for bin j, at channel k
    double _PWV;

    double* eta_atm_interp = new double[f_atm.num];
    double* PSD_nu = new double[f_atm.num];
    

    for(int i=start; i<stop; i++) {
        _PWV = PWV_atm.start + i * dPWV;
        for(int j=0; j<instrument->f_spec.num; j++) { 
            freq = instrument->f_spec.start + instrument->f_spec.step * j;

            eta_atm_interp[j] = interpValue(_PWV, freq, 
                    PWV_atm, f_atm, eta_atm);
            
            PSD_nu[j] = ( effs->eta_tot_chain * (1 - eta_atm_interp[j]) * I_atm[j] 
                + effs->eta_tot_gnd * I_gnd[j] 
                + effs->eta_tot_mir * I_tel[j]) 
                * CL*CL / (freq*freq);
        }
        
        // In this loop, calculate P_k, NEP_k and noise
        for(int k=0; k<instrument->nf_ch; k++) {
            double P_k = 0; // Initialise each channel to zero, for each timestep
            double eta_atm_avg = 0;
            double eta_kj_accum = 0;

            // Can loop over bins again, cheap operations this time
            for(int j=0; j<instrument->f_spec.num; j++) { 
                freq = instrument->f_spec.start + instrument->f_spec.step * j;
                eta_kj = instrument->filterbank[k * instrument->f_spec.num + j];
                
                eta_atm_avg += eta_atm_interp[j] * eta_kj;
                eta_kj_accum += eta_kj;
                P_k += PSD_nu[j] * eta_kj;
            }

            // STORAGE: Add signal to signal array in output
            //printf("%d\n", nPWV*k + i);
            output->power[k * nPWV + i] = P_k * instrument->f_spec.step; 
            output->temperature[k * nPWV + i] = atmosphere->Tatm * (1 - eta_atm_avg/eta_kj_accum); 
            //printf("%d\n", nPWV*k + i);
        }
    }
    delete[] eta_atm_interp;
    delete[] PSD_nu;
}
