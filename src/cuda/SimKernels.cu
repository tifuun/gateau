#include "InterfaceCUDA.h"

/*! \file Kernels.cu
    \brief Definitions of CUDA kernels for gateau.

    author: Arend Moerman
*/


// DEFINITIONS OF PHYSICAL AND MATHEMATICAL CONSTANTS
#define KB              1.380649E-23f
#define CL              2.9979246E8f
#define HP              6.62607015E-34f
#define PI              3.14159265

// HANDY STUFF
#define DEG2RAD PI/180

// OBSERVATION-INSTRUMENT PARAMETERS
__constant__ float cdt;                     // Timestep
__constant__ float cf_sample;               // Sampling frequency of readout
__constant__ float cGR_factor;              // Factor for GR noise: 2 * Delta / eta_pb
__constant__ int cnt;                       // Number of time evals
__constant__ int cnf_ch;                    // Number of filter freqs
__constant__ int cnum_stage;

// ATMOSPHERE PARAMETERS
__constant__ float ch_column;               // Column height
__constant__ float cv_wind;                 // Windspeed

// TEXTURE MEMORY
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_filterbank;
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_eta_ap;
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_psd_atm;

// CONSTANTS FOR KERNEL LAUNCHES
#define NTHREADS1D      256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/////////////////////////////////
//////// HOST FUNCTIONS /////////
/////////////////////////////////

/**
  Check CUDA API error status of call.
 
  Wrapper for finding errors in CUDA API calls.
 
  @param code The errorcode returned from failed API call.
  @param file The file in which failure occured.
  @param line The line in file in which error occured.
  @param abort Exit code upon error.
 */
__host__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
  Write a CUDA device array to a file, for debugging.
  Templated to work with floats and ints.

  @param array Pointer to device array of type T.
  @param s_array Size of array.
  @param name_txt Name of file to write array to. Name is appended with '.txt' by the function itself.
 */
template <typename T>
__host__ void writeArray(T *array, int s_array, std::string name_txt) {
    
    T *h_array = new T[s_array];
    gpuErrchk( cudaMemcpy(h_array, array, s_array * sizeof(T), cudaMemcpyDeviceToHost) );
    
    std::ofstream myfile (name_txt + ".txt");
    if (myfile.is_open())
    {
        for(int count = 0; count < s_array; count ++){
            myfile << h_array[count] << "\n" ;
        }

        myfile.close();
    }
    else std::cout << "Unable to open file";
    delete[] h_array;
}

__host__ inline float get_jn_noise(float T, float nu) 
{
    return HP * nu / (expf(HP * nu / (KB * T)) - 1);
}

/////////////////////////////////
/////// DEVICE FUNCTIONS ////////
/////////////////////////////////

/**
  Cascade a PSD through a reflector system, and couple to a specific parasitic PSD.

  @param P_nu_in PSD of incoming signal to be cascaded.
  @param eta Efficiency term associated with cascade.
  @param T_parasitic Temperature of parasitic source.
  @param nu Frequency in Hz.

  @returns Cascade output PSD.
 */
__device__ __inline__ float rad_trans(float psd_in, 
                                      float eta, 
                                      float psd_parasitic)
{
    return eta * psd_in + (1 - eta) * psd_parasitic;
}

/**
  Initialize CUDA.
 
  Instantiate program and populate constant memory.
 
  @param instrument CuInstrument object containing instrument to be simulated.
  @param telescope CuTelescope object containing telescope to be simulated.
  @param source CuSource object containing source definitions.
  @param atmosphere CuAtmosphere object containing atmosphere parameters.
  @param nTimes number of time evaluations in simulation.

  @return BT Array of two dim3 objects, containing number of blocks per grid and number of threads per block.
 */
__host__ void initCUDA(Instrument *instrument, 
        Telescope *telescope, 
        Source *source, 
        Atmosphere *atmosphere, 
        int nTimes,
        int num_stage) 
{
    float dt = 1. / instrument->f_sample;
    float GR_factor = 2 * instrument->delta / instrument->eta_pb;
     
    // OBSERVATION-INSTRUMENT PARAMETERS
    gpuErrchk( cudaMemcpyToSymbol(cdt, &dt, sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cf_sample, &(instrument->f_sample), sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cGR_factor, &GR_factor, sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cnt, &nTimes, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(cnf_ch, &(instrument->nf_ch), sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(cnum_stage, &num_stage, sizeof(int)) );
    
    // ATMOSPHERE PARAMETERS
    gpuErrchk( cudaMemcpyToSymbol(ch_column, &(atmosphere->h_column), sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cv_wind, &(atmosphere->v_wind), sizeof(float)) );
}

/**
  Main simulation kernel. This is where the magic happens.

  @param eta_cascade Array containing the transmission efficiency of each stage in the cascadei, including the final filterbank stage.
  @param psd_cascade Array containing the parasitic power spectral density of each stage in the cascade, including the final filterbank stage.
  @param num_stages Number of cascade stages, excluding the initial pass of the source signal through the atmosphere and the final filterbank stage.
  @param sigout Array for storing output power, for each channel, for each time, in SI units.
  @param azout Array containing Azimuth coordinates as function of time.
  @param elout Array containing Elevation coordinates as function of time.
  @param flagout Array for storing wether beam is in chop A or B, in nod AB or BA.
  @param PWV_trace Array containing PWV value of atmosphere as seen by telescope over observation, in millimeters.
  @param eta_atm Array with transmission parameters as function of freqs_atm and PWV_atm.
  @param source Array containing source intensity, as function of azsrc, elsrc and freqs_src, in SI units.
 */
__global__ void calcPowerNEP(ArrSpec f_src, 
                            float *az_scan, 
                            float *el_scan, 
                            float *PWV_screen,
                            ArrSpec x_atm,
                            ArrSpec y_atm,
                            ArrSpec f_atm, 
                            ArrSpec PWV_atm, 
                            ArrSpec az_src, 
                            ArrSpec el_src,
                            float *eta_cascade,
                            float *psd_cascade,
                            float *eta_atm,
                            float *sigout, 
                            float *nepout, 
                            float *source,
                            unsigned long long int seed = 0
                            ) 
{
    

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx < cnt) {
        ///////////////////////////////////////
        // DEFINITIONS OF REGISTER VARIABLES //
        ///////////////////////////////////////
        // FLOATS
        float I_nu;             // Specific intensity of source.
        float time;             // Timepoint for thread in simulation.
        float t, u;             // Interpolation factors
        float eta_atm_interp;   // Interpolated eta_atm, over frequency and PWV
        float freq;             // Bin frequency
        float psd_in;           // Local variable for storing PSD.
        float eta_kj;           // Filter efficiency for bin j, at channel k.
        float PWV_tr;           // Local variable for storing PWV value at time.
        float eta_ap;           // Local variable for storing aperture efficiency
        float sigfactor;        // Factor for calculating power. Perform outside of channel loop for speed.
        float csc_el;           // Cosecant of elevation angle.
        float psd_parasitic_use;
        float az_point, el_point;
        float x_point, y_point;
        float psd_atm;

        // INTEGERS
        int x0y0, x1y0, x0y1, x1y1; // Indices for interpolation
            
        curandState state;

        if (!seed) {
            seed = clock64();
        }

        seed += idx; // Make seed unique but reproduceable for each timestep, if seed is given.

        time = idx * cdt;

        az_point = az_scan[idx];
        el_point = el_scan[idx];

        x_point = __tanf(DEG2RAD * az_point) * ch_column + cv_wind * time;
        y_point = __tanf(DEG2RAD * el_point) * ch_column;

        // Interpolate on atmosphere
        interpValue(x_point, y_point,
                    &x_atm, &y_atm, PWV_screen, 0, PWV_tr);            
    
        curand_init(seed, idx, 0, &state);
        
        csc_el = 1. / __sinf(DEG2RAD * el_point);


        int iAz = floorf((az_point - az_src.start) / az_src.step);
        int iEl = floorf((el_point - el_src.start) / el_src.step);

        float az_src_max = az_src.start + az_src.step * (az_src.num - 1);
        float el_src_max = el_src.start + el_src.step * (el_src.num - 1);

        bool offsource = ((az_point < az_src.start) or (az_point > az_src_max)) or 
                         ((el_point < el_src.start) or (el_point > el_src_max));

        // This can be improved quite alot...
        if(offsource) 
        {
            az_point = az_src_max;
            el_point = el_src_max;
        }
        
        x0y0 = f_src.num * (iAz + iEl * az_src.num);
        x1y0 = f_src.num * (iAz + 1 + iEl * az_src.num);
        x0y1 = f_src.num * (iAz + (iEl+1) * az_src.num);
        x1y1 = f_src.num * (iAz + 1 + (iEl+1) * az_src.num);
        
        t = (az_point - (az_src.start + az_src.step*iAz)) / az_src.step;
        u = (el_point - (el_src.start + el_src.step*iEl)) / el_src.step;
        
        // Hier ongeveer starten met loopen over f_src
        for (int idy=0; idy<f_src.num; idy++)
        {
            //if (idx == 0) {
                //printf("%d, %d\n", idy, idx);
                //printf("%.12e, %.12e, %.12e\n", el_point, el_src.start, el_src_max);
            //}
            I_nu = (1-t)*(1-u) * source[x0y0 + idy];
            I_nu += t*(1-u) * source[x1y0 + idy];
            I_nu += (1-t)*u * source[x0y1 + idy];
            I_nu += t*u * source[x1y1 + idy];

            freq = f_src.start + f_src.step * idy;

            interpValue(PWV_tr, freq,
                        &PWV_atm, &f_atm,
                        eta_atm, 0, eta_atm_interp);

            eta_ap = tex1Dfetch(tex_eta_ap, idy); 

            eta_atm_interp = __powf(eta_atm_interp, csc_el);
            psd_atm = tex1Dfetch(tex_psd_atm, idy);

            // Initial pass through atmosphere
            psd_in = eta_ap * I_nu * CL*CL / (freq*freq); 
            psd_in = rad_trans(psd_in, eta_atm_interp, psd_atm);

            // Radiative transfer cascade
            #pragma unroll 
            for (int n=0; n<cnum_stage; n++) 
            {
                psd_parasitic_use = psd_cascade[idy + n*f_src.num];
                if (psd_parasitic_use < 0) 
                {
                    psd_parasitic_use = eta_atm_interp * psd_atm;
                }

                psd_in = rad_trans(psd_in, eta_cascade[idy + n*f_src.num], psd_parasitic_use);
            }

            float psd_filterbank = psd_cascade[idy + cnum_stage*f_src.num];
            float eta_filterbank = eta_cascade[idy + cnum_stage*f_src.num];

            #pragma unroll 
            for(int k=0; k<cnf_ch; k++) {
                eta_kj = tex1Dfetch( tex_filterbank, k*f_src.num + idy) * eta_filterbank;
                psd_in = rad_trans(psd_in, eta_kj, psd_filterbank);

                sigfactor = psd_in * f_src.step; // Note that psd_in already has the eta_kj incorporated!

                sigout[k*cnt + idx] += sigfactor; 
                nepout[k*cnt + idx] += sigfactor * (HP * freq + eta_kj * psd_in + cGR_factor); 
            }
        }
        
        float sqrt_samp = sqrtf(0.5 * cf_sample); // Constant term needed for noise calculation
        float sigma_k, P_k;

        #pragma unroll 
        for(int k=0; k<cnf_ch; k++) {
            sigma_k = sqrtf(2 * nepout[k*cnt + idx]) * sqrt_samp;
            P_k = sigma_k * curand_normal(&state);

            sigout[k*cnt + idx] += P_k;
        }
    }
}

/**
  Run a gateau simulation.
 
  This function is exposed to the ctypes interface and can be called from Python..
 
  @param instrument CuInstrument object containing instrument to be simulated.
  @param telescope CuTelescope object containing telescope to be simulated.
  @param atmosphere CuAtmosphere object containing atmosphere parameters.
  @param source CuSource object containing source definitions.
  @param nTimes Number of time evaluations in simulation.
 */
void run_gateau(Instrument *instrument, 
                     Telescope *telescope, 
                     Atmosphere *atmosphere, 
                     Source *source, 
                     Cascade *cascade,
                     int nTimesTotal, 
                     char *outpath) {
    // FLOATS
    float *d_sigout;        // Device pointer for output power array
    float *d_nepout;        // Device pointer for output power array
    float *d_I_nu;          // Device pointer for source intensities
    
    // INTEGERS
    int nffnt;              // Number of filter frequencies times number of time evaluations
    int nf_src;             // Number of frequency points in source.
    int numSMs;             // Number of streaming multiprocessors on GPU
    int nBlocks1D;          // Number of 1D blocks, in terms of number of SMs

    // OTHER DECLARATIONS
    dim3 blockSize1D;       // Size of 1D block (same as nThreads1D, but dim3 type)
    dim3 gridSize1D;        // Number of 1D blocks per grid

    // ALLOCATE ARRAY SPECIFICATION COPIES
    struct ArrSpec _f_spec = source->f_spec;
    struct ArrSpec _Az_src = source->az_src_spec;
    struct ArrSpec _El_src = source->el_src_spec;
    
    struct ArrSpec _f_atm;
    struct ArrSpec _PWV_atm;
    float *eta_atm;

    readEtaATM<float, ArrSpec>(&eta_atm, &_PWV_atm, &_f_atm);
    
    std::string str_path(atmosphere->path);
    std::string str_outpath(outpath);

    int *meta;
    readAtmMeta(&meta, str_path);

    // Calculate lengths of x and y of single screen
    float lx = meta[1]*atmosphere->dx;              // Length of a single screen along x, in meters
    float ly = meta[2]*atmosphere->dy;              // Length of a single screen along y, in meters
    float lx_av = lx - ly;                          // Available length along x, taking into account center of screen
    float t_obs_av = lx_av / atmosphere->v_wind;    // Max available time per screen

    float timeTotal = nTimesTotal / instrument->f_sample;       // Total time required for simulation

    int nJobs = ceil(timeTotal / t_obs_av);                     // Total number of times kernel needs to be run
    int nTimesScreen = floor(t_obs_av * instrument->f_sample);  // Number of time evaluations available per atmosphere screen. Floored to be safe.

    struct ArrSpec _x_atm;
    struct ArrSpec _y_atm;

    _x_atm.start = -ly/2;
    _x_atm.step = atmosphere->dx;
    _x_atm.num = meta[1];
    
    _y_atm.start = -ly/2;
    _y_atm.step = atmosphere->dy;
    _y_atm.num = meta[2];

    // Initialize constant memory
    initCUDA(instrument, telescope, source, atmosphere, nTimesScreen, cascade->num_stage); 

    nf_src = _f_spec.num; // Number of spectral points in source
    
    gpuErrchk( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0) );

    // TiEMPO2 prefers larger L1 cache over shared memory.
    gpuErrchk( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
    
    float freq;    // Frequency, used for initialising background sources.

    // Allocate cascade arrays
    std::vector<float> psd_atm(nf_src);

    for(int j=0; j<nf_src; j++)
    {
        freq = _f_spec.start + _f_spec.step * j;
        
        psd_atm[j] = get_jn_noise(atmosphere->T_atm, freq); 
    }
    
    float *d_psd_atm;

    gpuErrchk( cudaMalloc((void**)&d_psd_atm, nf_src * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_psd_atm, psd_atm.data(), nf_src * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaBindTexture((size_t)0, tex_psd_atm, d_psd_atm, nf_src * sizeof(float)) );
    
    // Allocate cascade arrays
    float *deta_cascade, *dpsd_cascade;
    gpuErrchk( cudaMalloc((void**)&deta_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&dpsd_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float)) );
    gpuErrchk( cudaMemcpy(deta_cascade, cascade->eta_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dpsd_cascade, cascade->psd_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    
    // Allocate and copy telescope arrays
    float *deta_ap;
    gpuErrchk( cudaMalloc((void**)&deta_ap, nf_src * sizeof(float)) );
    gpuErrchk( cudaMemcpy(deta_ap, telescope->eta_ap, nf_src * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaBindTexture((size_t)0, tex_eta_ap, deta_ap, nf_src * sizeof(float)) );

    float *daz_scan, *del_scan;
    gpuErrchk( cudaMalloc((void**)&daz_scan, nTimesTotal * sizeof(float)) );
    gpuErrchk( cudaMemcpy(daz_scan, telescope->az_scan, nTimesTotal * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void**)&del_scan, nTimesTotal * sizeof(float)) );
    gpuErrchk( cudaMemcpy(del_scan, telescope->el_scan, nTimesTotal * sizeof(float), cudaMemcpyHostToDevice) );

    // Allocate and copy atmosphere arrays
    float *deta_atm;
    int neta_atm = _f_atm.num * _PWV_atm.num;
    
    gpuErrchk( cudaMalloc((void**)&deta_atm, neta_atm * sizeof(float)) );
    gpuErrchk( cudaMemcpy(deta_atm, eta_atm, neta_atm * sizeof(float), cudaMemcpyHostToDevice) );
    delete[] eta_atm;

    // Allocate and copy instrument arrays
    float *dfilterbank;
    int nfilterbank = nf_src * instrument->nf_ch;
    gpuErrchk( cudaMalloc((void**)&dfilterbank, nfilterbank * sizeof(float)) );
    gpuErrchk( cudaMemcpy(dfilterbank, instrument->filterbank, nfilterbank * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaBindTexture((size_t)0, tex_filterbank, dfilterbank, nfilterbank * sizeof(float)) );
    
    gpuErrchk( cudaMalloc((void**)&d_I_nu, source->nI_nu * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_I_nu, source->I_nu, source->nI_nu * sizeof(float), cudaMemcpyHostToDevice) );

    std::string datp;

    int size_heap = 1024 * 2 * instrument->nf_ch * sizeof(float);

    // Loop starts here
    printf("\033[92m");
    int idx_wrap = 0;
    int time_counter = 0;
    for(int idx=0; idx<nJobs; idx++) {
        if (idx_wrap == meta[0]) {
            idx_wrap = 0;
        }

        if (idx == (nJobs - 1)) {
            nTimesScreen = nTimesTotal - nTimesScreen*(nJobs-1);
        }
        time_counter += nTimesScreen;

        printf("*** Progress: %d / 100 ***\r", time_counter*100 / nTimesTotal);
        fflush(stdout);

        nffnt = instrument->nf_ch * nTimesScreen; // Number of elements in single-screen output.
        gpuErrchk( cudaMemcpyToSymbol(cnt, &nTimesScreen, sizeof(int)) );
        
        nBlocks1D = ceilf((float)nTimesScreen / NTHREADS1D / numSMs);
        blockSize1D = NTHREADS1D;
        gridSize1D = nBlocks1D*numSMs;

        // Allocate output arrays
        gpuErrchk( cudaMalloc((void**)&d_sigout, nffnt * sizeof(float)) );
        gpuErrchk( cudaMalloc((void**)&d_nepout, nffnt * sizeof(float)) );

        // Allocate PWV screen now, delete CUDA allocation after first kernel call
        float *PWV_screen;
        float *dPWV_screen;
        
        int nPWV_screen = _x_atm.num * _y_atm.num;
        
        //curandState *devStates;
        //gpuErrchk( cudaMalloc((void **)&devStates, nTimesScreen * sizeof(curandState)) );

        datp = std::to_string(idx_wrap) + ".datp";
        readAtmScreen<float, ArrSpec>(&PWV_screen, &_x_atm, &_y_atm, str_path, datp);
        
        gpuErrchk( cudaMalloc((void**)&dPWV_screen, nPWV_screen * sizeof(float)) );
        gpuErrchk( cudaMemcpy(dPWV_screen, PWV_screen, nPWV_screen * sizeof(float), cudaMemcpyHostToDevice) );
       
        // CALL TO MAIN SIMULATION KERNEL
        calcPowerNEP<<<gridSize1D, blockSize1D>>>(_f_spec, 
                                                  daz_scan,
                                                  del_scan,
                                                  dPWV_screen,
                                                  _x_atm,
                                                  _y_atm,
                                                  _f_atm, 
                                                  _PWV_atm, 
                                                  _Az_src, 
                                                  _El_src, 
                                                  deta_cascade,
                                                  dpsd_cascade, 
                                                  deta_atm,
                                                  d_sigout,
                                                  d_nepout,
                                                  d_I_nu);
        
        gpuErrchk( cudaDeviceSynchronize() );
        
        //gpuErrchk( cudaFree(devStates) );
        gpuErrchk( cudaFree(dPWV_screen) );
        
        // ALLOCATE STRINGS FOR WRITING OUTPUT
        std::string signame = std::to_string(idx) + "signal.out";

        std::vector<float> sigout(nffnt);

        gpuErrchk( cudaMemcpy(sigout.data(), d_sigout, nffnt * sizeof(float), cudaMemcpyDeviceToHost) );

        write1DArray<float>(sigout, str_outpath, signame);
        
        gpuErrchk( cudaFree(d_sigout) );
        gpuErrchk( cudaFree(d_nepout) );

        idx_wrap++;
    }
    gpuErrchk( cudaDeviceReset() );
    printf("\033[0m\n");
}

