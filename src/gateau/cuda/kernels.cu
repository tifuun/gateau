#include "interface.h"


#include <fitsio.h>

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
__constant__ float ct_start;                // Starting time
__constant__ float cf_sample;               // Sampling frequency of readout
__constant__ float csqrt_samp;               // Sampling frequency of readout
__constant__ float cGR_factor;              // Factor for GR noise: 2 * Delta / eta_pb
__constant__ int cnt;                       // Number of time evals
__constant__ int cnf_ch;                    // Number of filter freqs
__constant__ int cnum_stage;

// ATMOSPHERE PARAMETERS
__constant__ float ch_column;               // Column height
__constant__ float cv_wind;                 // Windspeed
__constant__ float cpwv0;

// CONSTANTS FOR KERNEL LAUNCHES
#define NTHREADS1D      512

#define NTHREADS2DX     32
#define NTHREADS2DY     16

#define MEMBUFF         0.8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL

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

__device__ __inline__ void time_wrt_to(int thread_index, 
                                                int thread_index_select=0,
                                                long long int time_offset=0)
{
    if(thread_index == thread_index_select) 
    {
        printf("Thread %d at time %llu w.r.t. offset\n", thread_index, clock64() - time_offset);
    }
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
    float sqrt_samp = sqrtf(0.5 / dt); // Constant term needed for noise calculation

    // OBSERVATION-INSTRUMENT PARAMETERS
    gpuErrchk( cudaMemcpyToSymbol(cdt, &dt, sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cf_sample, &(instrument->f_sample), sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(csqrt_samp, &sqrt_samp, sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cGR_factor, &GR_factor, sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cnt, &nTimes, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(cnf_ch, &(instrument->nf_ch), sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(cnum_stage, &num_stage, sizeof(int)) );

    // ATMOSPHERE PARAMETERS
    gpuErrchk( cudaMemcpyToSymbol(ch_column, &(atmosphere->h_column), sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cv_wind, &(atmosphere->v_wind), sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cpwv0, &(atmosphere->pwv0), sizeof(float)) );
}

/**
  Initialize random number generator state.

 */
__global__ void init_random_states(curandState *state,
                                   unsigned long long int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                 
                                                                                     
    if (idx < cnt) 
    {
        if (!seed)
        {
            seed = clock64();
        }
        curand_init(seed, idx, 0, &state[idx]);                                      
    }
}

__global__ void calc_onef_psd(cufftComplex *output,
                              float *onef_level,
                              float *onef_conv,
                              float *onef_alpha,
                              curandState *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                 
                                                                                     
    if (idx < (cnt / 2)) 
    {
        cufftComplex local;
        curandState localState = state[idx];
        
        if (!idx) {
            local.x = 0.;
            local.y = 0.;
        }

        else {
            //float factor = 1 / (4 * PI * idx * cf_sample / 2 / (cnt / 2 + 1));
            float factor = 1 / (idx * cf_sample / 2 / (cnt / 2 + 1));
            float factor_loc;

            for(int k=0; k<cnf_ch; k++) {
                factor_loc = sqrtf(onef_level[k] / cnt) * onef_conv[k] * powf(factor, onef_alpha[k]/2);
                local.x = factor_loc * curand_normal(&localState);
                local.y = factor_loc * curand_normal(&localState);
                output[k*(cnt / 2 + 1) + idx] = local;
            }
            state[idx] = localState;
        }
    }
}

__global__ void calc_traces_rng(float *az_scan, 
                                float *el_scan,
                                float *az_scan_center, 
                                float *el_scan_center,
                                float az_fpa,
                                float el_fpa,
                                ArrSpec x_atm,
                                ArrSpec y_atm,
                                float *pwv_screen,
                                float *az_trace,
                                float *el_trace,
                                float *pwv_trace,
                                float *time_trace,
                                unsigned int idx_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                 
                                                                                     
    if (idx < cnt) 
    {
        // FLOATS                                                                    
        float time_point;  // Timepoint for thread in simulation. This is not the global time, only time in current simulation chunk!                   
        float pwv_point;   // Container for storing interpolated PWV values.        
        float az_point, el_point;
        float x_point, y_point;
                                                                                     
        time_point = idx * cdt;

        az_point = az_scan[idx + idx_offset] + az_fpa;
        el_point = el_scan[idx + idx_offset] + el_fpa;

        x_point = __tanf(DEG2RAD * (az_point - az_scan_center[idx + idx_offset])) * ch_column + cv_wind * time_point;
        y_point = __tanf(DEG2RAD * (el_point - el_scan_center[idx + idx_offset])) * ch_column;

        interpValue(x_point, y_point,
                    &x_atm, &y_atm, pwv_screen, 0, pwv_point);            
                                                                                     
        az_trace[idx] = az_point;                                                    
        el_trace[idx] = el_point;
        pwv_trace[idx] = pwv_point + cpwv0;

        time_trace[idx] = time_point + ct_start;
    }
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
__global__ void calc_power(float *az_trace, 
                            float *el_trace, 
                            float *pwv_trace,
                            ArrSpec f_atm, 
                            ArrSpec pwv_atm, 
                            ArrSpec az_src, 
                            ArrSpec el_src,
                            ArrSpec f_src,
                            float *eta_ap,
                            float *psd_atm,
                            float *eta_cascade,
                            float *psd_cascade,
                            float *eta_atm,
                            float *filterbank,
                            float *sigout, 
                            float *nepout, 
                            float *source)
{
    

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int idy = blockIdx.y * blockDim.y + threadIdx.y; 

    __shared__ float csc_el_shared[NTHREADS2DX];

    if (idx < cnt && idy < f_src.num) {
        ///////////////////////////////////////
        // DEFINITIONS OF REGISTER VARIABLES //
        ///////////////////////////////////////
        // FLOATS
        float psd_nu;             // Specific intensity of source.
        float t, u;             // Interpolation factors
        float eta_atm_interp;   // Interpolated eta_atm, over frequency and PWV
        float freq;             // Bin frequency
        float psd_in;           // Local variable for storing PSD.
        float psd_in_k;         // Local variable for calculating psd per channel
        float eta_kj;           // Filter efficiency for bin j, at channel k.
        float eta_ap_loc;       // Local variable for storing aperture efficiency
        float sigfactor;        // Factor for calculating power. Perform outside of channel loop for speed.
        float psd_parasitic_use;
        float temp1, temp2, temp3;
        float psd_atm_loc;

        // INTEGERS
        int x0y0, x1y0, x0y1, x1y1; // Indices for interpolation

        temp1 = az_trace[idx];
        temp2 = el_trace[idx];
        temp3 = pwv_trace[idx];

        if(threadIdx.y == 0) 
        {
            csc_el_shared[threadIdx.x] = 1. / __sinf(DEG2RAD * temp2);
        }

        __syncthreads();

        int iaz = floorf((temp1 - az_src.start) / az_src.step);
        int iel = floorf((temp2 - el_src.start) / el_src.step);

        float az_src_max = az_src.start + az_src.step * (az_src.num - 1);
        float el_src_max = el_src.start + el_src.step * (el_src.num - 1);

	// THis line used to have `or` instead of `||` but apparently
	// Bill Gates says this is a no-no
        bool offsource = ((temp1 < az_src.start) || (temp1 > az_src_max)) || 
                         ((temp2 < el_src.start) || (temp2 > el_src_max));

        // This can be improved quite alot...
        if(offsource) 
        {
            temp1 = az_src_max;
            temp2 = el_src_max;
        }
        
        x0y0 = f_src.num * (iaz + iel * az_src.num);
        x1y0 = f_src.num * (iaz + 1 + iel * az_src.num);
        x0y1 = f_src.num * (iaz + (iel+1) * az_src.num);
        x1y1 = f_src.num * (iaz + 1 + (iel+1) * az_src.num);
        
        t = (temp1 - (az_src.start + az_src.step*iaz)) / az_src.step;
        u = (temp2 - (el_src.start + el_src.step*iel)) / el_src.step;
        
        psd_nu = (1-t)*(1-u) * source[x0y0 + idy];
        psd_nu += t*(1-u) * source[x1y0 + idy];
        psd_nu += (1-t)*u * source[x0y1 + idy];
        psd_nu += t*u * source[x1y1 + idy];

        freq = f_src.start + f_src.step * idy;

        interpValue(temp3, freq,
                    &pwv_atm, &f_atm,
                    eta_atm, 0, eta_atm_interp);

        eta_ap_loc = eta_ap[idy]; 

        eta_atm_interp = __powf(eta_atm_interp, csc_el_shared[threadIdx.x]);
        psd_atm_loc = psd_atm[idy];

        // Initial pass through atmosphere
        psd_in = eta_ap_loc * psd_nu * 0.5; 
        
        psd_in = rad_trans(psd_in, eta_atm_interp, psd_atm_loc);

        // Radiative transfer cascade
        #pragma unroll 
        for (int n=0; n<cnum_stage; n++) 
        {
            psd_parasitic_use = psd_cascade[n*f_src.num + idy];
            if (psd_parasitic_use < 0) 
            {
                psd_parasitic_use = eta_atm_interp * psd_atm_loc;
            }

            psd_in = rad_trans(psd_in, eta_cascade[n*f_src.num + idy], psd_parasitic_use);
        }

        temp1 = eta_cascade[cnum_stage*f_src.num + idy];
        temp2 = psd_cascade[cnum_stage*f_src.num + idy];

        #pragma unroll 
        for(int k=0; k<cnf_ch; k++) {
            eta_kj = filterbank[k*f_src.num + idy] * temp1;
            psd_in_k = rad_trans(psd_in, eta_kj, temp2);

            sigfactor = psd_in_k * f_src.step; // Note that psd_in already has the eta_kj incorporated!

            atomicAdd(&sigout[k*cnt + idx], sigfactor); 
            atomicAdd(&nepout[k*cnt + idx], sigfactor * (HP * freq + eta_kj * psd_in_k + cGR_factor)); 
        }
    }
}

/**
  Calculate the total photon noise in a filter channel.

  After calculating the noise std from the NEP, a random number from a Gaussian is drawn and added to the total power in a channel.
  Note that, because we do not need the NEP after this step, we replace the value with a random Gaussian. 
  This is necessary for the TLS noise calculation, which comes after.

  @param sigout Array for storing output power, for each channel, for each time, in SI units.
  @param nepout Array for storing output NEP, for each channel, for each time, in SI units.
  @param state Array with states for drawing random Gaussian values for noise calculations.
 */
__global__ void calc_photon_noise(float *sigout, 
                                 float *nepout, 
                                 float *c0,
                                 float *c1,
                                 curandState *state,
                                 unsigned int idx_in_screen) 
{    
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (idx < cnt) {
        curandState localState = state[idx + idx_in_screen];
        float P_k, sigma_k;
        float c0_loc, c1_loc;

        for(int k=0; k<cnf_ch; k++) {
            c0_loc = c0[k];
            c1_loc = c1[k];
            sigma_k = sqrtf(2 * nepout[k*cnt + idx]) * csqrt_samp;
            P_k = sigout[k*cnt + idx] + sigma_k * curand_normal(&localState);
            sigout[k*cnt + idx] = c0_loc + c1_loc * P_k;

        }
        state[idx + idx_in_screen] = localState;
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
                 int nttot, 
                 char *outpath,
                 char *outscale,
                 unsigned long long int seed,
                 char *atmpath
		 ) 
{
	// TESTING CFITSIO

    fitsfile *fptr;
    int status=0;
    fits_open_file(&fptr, "tq123x.kjl", READWRITE, &status);
    printf("  ffopen fptr, status  = %lu %d (expect an error)\n", 
           (unsigned long) fptr, status);
    ffclos(fptr, &status);

    // DEVICE POINTERS: FLOATS
    float *d_sigout;        // Output power/temperature array
    float *d_nepout;        // Output NEP array
    float *d_I_nu;          // Source intensities
    float *d_az_trace;      // Azimuth traces
    float *d_el_trace;      // Elevation traces
    float *d_time_trace;    // Time traces
    float *d_pwv_trace;     // PWV traces
    
    // HOST: INTEGERS
    unsigned long long nf_ch;              // Number of channels per spaxel
    unsigned long long nf_src;             // Number of frequency points in source
    unsigned long long nffnt;              // Number of channels times number of time evaluations
    unsigned long long ntscr;              // Number of time points per atmosphere screen

    unsigned long long nf_sub_fnt;         // Number of channels times number of time evaluations per subblock
    unsigned long long nt_sub_scr;         // Number of time points per subblock, inside atmosphere screen
    unsigned long long npwvscr;

    // HOST: KERNEL LAUNCH SPECS
    int numSMs;             // Number of streaming multiprocessors on GPU
    int nBlocks1D;          // Number of 1D blocks, in terms of number of SMs
    int nBlocks2Dx;         // Number of 2D blocks along time axis, in terms of number of SMs
    int nBlocks2Dy;         // Number of 2D blocks along channel axis, in terms of number of SMs
    dim3 blockSize1D;       // Size of 1D block (same as nThreads1D, but dim3 type)
    dim3 gridSize1D;        // Number of 1D blocks per grid
    dim3 blockSize2D;       // Size of 2D block
    dim3 gridSize2D;        // Number of 2D blocks per grid

    // ALLOCATE ARRAY SPECIFICATION COPIES
    struct ArrSpec f_src = source->f_spec;
    struct ArrSpec az_src = source->az_src_spec;
    struct ArrSpec el_src = source->el_src_spec;
    
    struct ArrSpec f_atm;
    struct ArrSpec pwv_atm;
    float *eta_atm;

    const auto processor_count = std::thread::hardware_concurrency();

    char *power_str = "P";
    char *tb_str = "Tb";

    readEtaATM<float, ArrSpec>(&eta_atm, &pwv_atm, &f_atm, atmpath);
    
    std::string str_path(atmosphere->path);
    std::string str_outpath(outpath);

    int *meta;
    readAtmMeta(&meta, str_path);
    
    float ttot = nttot / instrument->f_sample;       // Total time required for simulation

    // Calculate lengths of x and y of single screen
    float lx = meta[1]*atmosphere->dx;              // Length of a single screen along x, in meters
    float ly = meta[2]*atmosphere->dy;              // Length of a single screen along y, in meters
    float lx_av = lx - ly;                          // Available length along x, taking into account center of screen
    float t_obs_av = lx_av / atmosphere->v_wind;    // Max available time per screen

    if(isinf(t_obs_av) || isnan(t_obs_av)) {t_obs_av = ttot;}

    int nJobs = ceil(ttot / t_obs_av);                     // Total number of times kernel needs to be run

    // Following int is number of evaluations per a
    ntscr = floor(t_obs_av * instrument->f_sample);  // Number of time evaluations available per atmosphere screen. Floored to be safe.

    struct ArrSpec x_atm;
    struct ArrSpec y_atm;

    x_atm.start = -ly/2;
    x_atm.step = atmosphere->dx;
    x_atm.num = meta[1];
    
    y_atm.start = -ly/2;
    y_atm.step = atmosphere->dy;
    y_atm.num = meta[2];

    // Initialize constant memory
    initCUDA(instrument, telescope, source, atmosphere, ntscr, cascade->num_stage); 

    nf_src = f_src.num; // Number of spectral points in source
    nf_ch = instrument->nf_ch;
    
    gpuErrchk( cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0) );

    // TiEMPO2 prefers larger L1 cache over shared memory.
    gpuErrchk( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
    
    float freq;    // Frequency, used for initialising background sources.

    // Allocate cascade arrays
    std::vector<float> psd_atm(nf_src);

    for(int j=0; j<nf_src; j++)
    {
        freq = f_src.start + f_src.step * j;
        
        psd_atm[j] = get_jn_noise(atmosphere->T_atm, freq); 
    }
    
    float *d_psd_atm;

    gpuErrchk( cudaMalloc((void**)&d_psd_atm, nf_src * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_psd_atm, psd_atm.data(), nf_src * sizeof(float), cudaMemcpyHostToDevice) );
    
    // Allocate cascade arrays
    float *d_eta_cascade, *d_psd_cascade;
    gpuErrchk( cudaMalloc((void**)&d_eta_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_psd_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_eta_cascade, cascade->eta_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_psd_cascade, cascade->psd_cascade, nf_src * (cascade->num_stage + 1) * sizeof(float), cudaMemcpyHostToDevice) );
    
    // Allocate and copy telescope arrays
    float *d_eta_ap;
    gpuErrchk( cudaMalloc((void**)&d_eta_ap, nf_src * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_eta_ap, telescope->eta_ap, nf_src * sizeof(float), cudaMemcpyHostToDevice) );

    float *d_az_scan, *d_el_scan;
    gpuErrchk( cudaMalloc((void**)&d_az_scan, nttot * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_az_scan, telescope->az_scan, nttot * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void**)&d_el_scan, nttot * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_el_scan, telescope->el_scan, nttot * sizeof(float), cudaMemcpyHostToDevice) );

    float *d_az_scan_center, *d_el_scan_center;
    gpuErrchk( cudaMalloc((void**)&d_az_scan_center, nttot * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_az_scan_center, telescope->az_scan_center, nttot * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void**)&d_el_scan_center, nttot * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_el_scan_center, telescope->el_scan_center, nttot * sizeof(float), cudaMemcpyHostToDevice) );
    // Allocate and copy atmosphere arrays
    float *d_eta_atm;
    int neta_atm = f_atm.num * pwv_atm.num;
    
    gpuErrchk( cudaMalloc((void**)&d_eta_atm, neta_atm * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_eta_atm, eta_atm, neta_atm * sizeof(float), cudaMemcpyHostToDevice) );

    // Perform responsivity calibration, if desired. 
    // Do now before we delete eta_atm!
    float *c0 = new float[nf_ch];
    float *c1 = new float[nf_ch];
    
    // Change this so that it takes a variable
    if(!strcmp(outscale, tb_str))
    {
        float *Psky = new float[nf_ch * NPWV]();
        float *Tsky = new float[nf_ch * NPWV]();
        float *eta_kj_sum = new float[nf_ch]();
        
        for(int k=0; k<nf_ch; k++)
        {
            for(int j=0; j<f_src.num; j++)
            {
                eta_kj_sum[k] += instrument->filterbank[k*f_src.num + j];
            }
        }
    
        // Make threadpool
        std::vector<std::thread> threadPool;
        threadPool.resize(processor_count);

        int step = ceil(NPWV / processor_count);

        for(int n=0; n < processor_count; n++) 
        {
            int final_step; // Final step for

            if(n == (processor_count - 1)) {
                final_step = NPWV;
            } else {
                final_step = (n+1) * step;
            }

            threadPool[n] = std::thread(&resp_calibration, 
                    n * step,
                    final_step,
                    &f_atm, 
                    &pwv_atm, 
                    &f_src,
                    atmosphere->T_atm, 
                    nf_ch,
                    cascade->eta_cascade, 
                    cascade->psd_cascade,
                    cascade->num_stage, 
                    psd_atm.data(), 
                    eta_atm,
                    instrument->filterbank, 
                    eta_kj_sum,
                    Psky, 
                    Tsky);
        }

        // Wait with execution until all threads are done
        for (std::thread &t : threadPool) {
            if (t.joinable()) {
                t.join();
            }
        }
        delete[] eta_kj_sum;

        fit_calibration(Psky,
                Tsky,
                nf_ch,
                c0,
                c1);
        delete[] Psky;   
        delete[] Tsky;   
    }

    if(!strcmp(outscale, power_str))
    {
        for(int i=0; i<nf_ch; i++)
        {
            c0[i] = 0.;
            c1[i] = 1.;
        }
    }

    float *d_c0, *d_c1;
    gpuErrchk( cudaMalloc((void**)&d_c0, nf_ch * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_c0, c0, nf_ch * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc((void**)&d_c1, nf_ch * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_c1, c1, nf_ch * sizeof(float), cudaMemcpyHostToDevice) );
    
    delete[] eta_atm;
    delete[] c0;
    delete[] c1;

    // Allocate and copy instrument arrays
    float *d_filterbank;
    int nfilterbank = nf_src * nf_ch;
    gpuErrchk( cudaMalloc((void**)&d_filterbank, nfilterbank * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_filterbank, instrument->filterbank, nfilterbank * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_I_nu, source->nI_nu * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_I_nu, source->I_nu, source->nI_nu * sizeof(float), cudaMemcpyHostToDevice) );

    std::string datp;

    // Loop starts here
    printf("\033[92m");
    int idx_wrap;
    unsigned int idx_offset;        // Index of time inside the total time
    unsigned int idx_in_screen;     // Index of time inside a single screen
    int idx_write; // Counter for serializing output chunks

    size_t free_mem, total_mem, required_mem;
    float free_required_frac;
    int num_blocks_in_screen;                   // Number of blocks within a single screen for memory requirements.

    int num_spax = instrument->num_spax;
    float az_fpa, el_fpa;

    float ftime_counter;
    
    curandState *devstates;
    gpuErrchk( cudaMalloc((void**)&devstates, ntscr * sizeof(curandState)) );
    
    nBlocks1D = ceilf((float)ntscr / NTHREADS1D / numSMs);
    blockSize1D = NTHREADS1D;
    gridSize1D = nBlocks1D*numSMs;
  
    init_random_states<<<gridSize1D, blockSize1D>>>(devstates, seed);

    int ntscr_job;
    int nt_sub_scr_job;
    for(int idx_spax=0; idx_spax<num_spax; idx_spax++) 
    {
        printf("Simulating spaxel %d / %d\n", idx_spax+1, num_spax);
        gpuErrchk( cudaMemcpyToSymbol(cnt, &ntscr, sizeof(int)) );
        az_fpa = instrument->az_fpa[idx_spax];
        el_fpa = instrument->el_fpa[idx_spax];

        idx_wrap = 0;
        idx_offset = 0;
        idx_write = 0;
        ftime_counter = 0.;

        // Current 1/f implementation could cause memory issues!

        for(int idx=0; idx<nJobs; idx++) {
            idx_in_screen = 0;
            ntscr_job = ntscr;

            if (idx_wrap == meta[0]) {
                idx_wrap = 0;
            }

            if (idx == (nJobs - 1)) {
                ntscr_job = nttot - ntscr * idx;
            }

            nffnt = nf_ch * ntscr_job; // Number of elements in single-screen output.
            
            // Allocate PWV screen now, delete CUDA allocation after first kernel call
            float *pwv_screen;
            float *d_pwv_screen;
            
            datp = std::to_string(idx_wrap) + ".datp";
            readAtmScreen<float, ArrSpec>(&pwv_screen, &x_atm, &y_atm, str_path, datp);
            
            npwvscr = x_atm.num * y_atm.num;

            // Currently, reads the entire screen even if only part of it will be processed in the next loop.
            gpuErrchk( cudaMalloc((void**)&d_pwv_screen, npwvscr * sizeof(float)) );
            gpuErrchk( cudaMemcpy(d_pwv_screen, pwv_screen, npwvscr * sizeof(float), cudaMemcpyHostToDevice) );

            // Allocate output arrays
            // Check how much free memory - if insufficient, loop again here
            gpuErrchk( cudaMemGetInfo(&free_mem, &total_mem) );

            free_mem *= MEMBUFF;

            required_mem = (4 * ntscr_job + 2 * nffnt + npwvscr) * sizeof(float);

            num_blocks_in_screen = (int)ceil((float)required_mem / free_mem);
            
            nt_sub_scr = (int)floor((float)ntscr_job / num_blocks_in_screen);

            for(int idx_sub=0; idx_sub<num_blocks_in_screen; idx_sub++)
            {
                // nt_sub_scr is really the amount of time evaluations per kernel launch here
                nt_sub_scr_job = nt_sub_scr;
                if (idx_sub == (num_blocks_in_screen - 1)) {
                    nt_sub_scr_job = ntscr_job - nt_sub_scr * idx_sub;
                }

                if(nt_sub_scr_job % 2){nt_sub_scr_job -= 1;}  // For 1/f calculation, easier if ntscr is odd
                
                nf_sub_fnt = nf_ch * nt_sub_scr_job;
                
                gpuErrchk( cudaMalloc((void**)&d_sigout, nf_sub_fnt * sizeof(float)) );
                gpuErrchk( cudaMemcpyToSymbol(cnt, &nt_sub_scr_job, sizeof(int)) );
                
                if(instrument->use_onef) {
                    int ntscr_h = nt_sub_scr_job / 2 + 1;
                    nBlocks1D = ceilf((float)(ntscr_h) / NTHREADS1D / numSMs);
                    blockSize1D = NTHREADS1D;
                    gridSize1D = nBlocks1D*numSMs;

                    float *d_onef_level, *d_onef_conv, *d_onef_alpha;
                    gpuErrchk( cudaMalloc((void**)&d_onef_level, nf_ch * sizeof(float)) );
                    gpuErrchk( cudaMalloc((void**)&d_onef_conv, nf_ch * sizeof(float)) );
                    gpuErrchk( cudaMalloc((void**)&d_onef_alpha, nf_ch * sizeof(float)) );
                    gpuErrchk( cudaMemcpy(d_onef_level, instrument->onef_level, nf_ch * sizeof(float), cudaMemcpyHostToDevice) );
                    gpuErrchk( cudaMemcpy(d_onef_conv, instrument->onef_conv, nf_ch * sizeof(float), cudaMemcpyHostToDevice) );
                    gpuErrchk( cudaMemcpy(d_onef_alpha, instrument->onef_alpha, nf_ch * sizeof(float), cudaMemcpyHostToDevice) );

                    cufftComplex *d_onef_out;
                    gpuErrchk( cudaMalloc((void**)&d_onef_out, ntscr_h*nf_ch * sizeof(cufftComplex)) );

                    calc_onef_psd<<<gridSize1D, blockSize1D>>>(d_onef_out, 
                            d_onef_level, 
                            d_onef_conv, 
                            d_onef_alpha,
                            devstates);
                    gpuErrchk( cudaDeviceSynchronize() );
                    
                    gpuErrchk( cudaFree(d_onef_level) );
                    gpuErrchk( cudaFree(d_onef_conv) );
                    gpuErrchk( cudaFree(d_onef_alpha) );

                    cufftHandle plan;
                    int rank = 1;
                    int n[] = { nt_sub_scr_job };
                    int istride = 1, ostride = 1;
                    int idist = ntscr_h, odist = nt_sub_scr_job;
                    int inembed[] = { 0 };
                    int onembed[] = { 0 };

                    CUFFT_CALL( cufftPlanMany(&plan, rank, n, 
                            inembed, istride, idist,
                            onembed, ostride, odist, CUFFT_C2R, nf_ch) );

                    CUFFT_CALL( cufftExecC2R(plan, d_onef_out, d_sigout) );
                    gpuErrchk( cudaDeviceSynchronize() );
                    gpuErrchk( cudaFree(d_onef_out) );

                    CUFFT_CALL( cufftDestroy(plan) );
                    gpuErrchk( cudaDeviceSynchronize() );
                }

                else {
                    gpuErrchk( cudaMemset(d_sigout, 0, nf_sub_fnt * sizeof(float)) );
                }
                
                gpuErrchk( cudaMalloc((void**)&d_az_trace, nt_sub_scr_job * sizeof(float)) );
                gpuErrchk( cudaMalloc((void**)&d_el_trace, nt_sub_scr_job * sizeof(float)) );
                gpuErrchk( cudaMalloc((void**)&d_pwv_trace, nt_sub_scr_job * sizeof(float)) );
                gpuErrchk( cudaMalloc((void**)&d_time_trace, nt_sub_scr_job * sizeof(float)) );
                gpuErrchk( cudaMalloc((void**)&d_nepout, nf_sub_fnt * sizeof(float)) );

                
                nBlocks1D = ceilf((float)nt_sub_scr_job / NTHREADS1D / numSMs);
                blockSize1D = NTHREADS1D;
                gridSize1D = nBlocks1D*numSMs;

                nBlocks2Dx = ceilf((float)nt_sub_scr_job / NTHREADS2DX / numSMs);
                nBlocks2Dy = ceilf((float)nf_src / NTHREADS2DY / numSMs);
                blockSize2D = dim3(NTHREADS2DX, NTHREADS2DY);
                gridSize2D = dim3(nBlocks2Dx*numSMs, nBlocks2Dy*numSMs);
                
                ftime_counter = static_cast<float>(idx_offset) / instrument->f_sample;

                printf("*** Progress: %d / 100 ***\r", idx_offset*100 / nttot);
                fflush(stdout);

                gpuErrchk( cudaMemset(d_nepout, 0, nf_sub_fnt * sizeof(float)) );
                
                gpuErrchk( cudaMemcpyToSymbol(ct_start, &ftime_counter, sizeof(float)) );
                gpuErrchk( cudaMemcpyToSymbol(cnt, &nt_sub_scr_job, sizeof(int)) );

                calc_traces_rng<<<gridSize1D, 
                                  blockSize1D>>>
                                      (d_az_scan,
                                       d_el_scan,
                                       d_az_scan_center,
                                       d_el_scan_center,
                                       az_fpa,
                                       el_fpa,
                                       x_atm,
                                       y_atm,
                                       d_pwv_screen,
                                       d_az_trace,
                                       d_el_trace,
                                       d_pwv_trace,
                                       d_time_trace, 
                                       idx_offset);
                gpuErrchk( cudaDeviceSynchronize() );

                // CALL TO MAIN SIMULATION KERNEL
                calc_power<<<gridSize2D, 
                             blockSize2D>>>
                                 (d_az_trace,
                                  d_el_trace,
                                  d_pwv_trace,
                                  f_atm, 
                                  pwv_atm, 
                                  az_src, 
                                  el_src,
                                  f_src,
                                  d_eta_ap,
                                  d_psd_atm,
                                  d_eta_cascade,
                                  d_psd_cascade, 
                                  d_eta_atm,
                                  d_filterbank,
                                  d_sigout,
                                  d_nepout,
                                  d_I_nu);
                
                gpuErrchk( cudaDeviceSynchronize() );

                calc_photon_noise<<<gridSize1D, 
                                    blockSize1D>>>
                                        (d_sigout, 
                                         d_nepout, 
                                         d_c0,
                                         d_c1,
                                         devstates,
                                         idx_in_screen);

                gpuErrchk( cudaDeviceSynchronize() );
                
                gpuErrchk( cudaFree(d_nepout) );
                gpuErrchk( cudaFree(d_pwv_trace) );
                
                // ALLOCATE STRINGS FOR WRITING OUTPUT
                std::string signame = std::to_string(idx_write) + "signal.out";
                std::string azname = std::to_string(idx_write) + "az.out";
                std::string elname = std::to_string(idx_write) + "el.out";

                // Only write time for first spaxel
                if(idx_spax == 0) 
                {
                    std::string timename = std::to_string(idx_write) + "time.out";
                    std::vector<float> timeout(nt_sub_scr_job);
                    gpuErrchk( cudaMemcpy(timeout.data(), d_time_trace, nt_sub_scr_job * sizeof(float), cudaMemcpyDeviceToHost) );
                    write1DArray<float>(timeout, str_outpath, timename);
                    gpuErrchk( cudaFree(d_time_trace) );
                }

                std::vector<float> sigout(nf_sub_fnt);
                std::vector<float> azout(nt_sub_scr_job);
                std::vector<float> elout(nt_sub_scr_job);

                gpuErrchk( cudaMemcpy(sigout.data(), d_sigout, nf_sub_fnt * sizeof(float), cudaMemcpyDeviceToHost) );
                gpuErrchk( cudaMemcpy(azout.data(), d_az_trace, nt_sub_scr_job * sizeof(float), cudaMemcpyDeviceToHost) );
                gpuErrchk( cudaMemcpy(elout.data(), d_el_trace, nt_sub_scr_job * sizeof(float), cudaMemcpyDeviceToHost) );

                write1DArray<float>(sigout, str_outpath, signame, std::to_string(idx_spax));
                write1DArray<float>(azout, str_outpath, azname, std::to_string(idx_spax));
                write1DArray<float>(elout, str_outpath, elname, std::to_string(idx_spax));
                gpuErrchk( cudaFree(d_sigout) );
                gpuErrchk( cudaFree(d_az_trace) );
                gpuErrchk( cudaFree(d_el_trace) );
                
                idx_write++;
                idx_offset += nt_sub_scr_job;
                idx_in_screen += nt_sub_scr_job;
            }
            gpuErrchk( cudaDeviceSynchronize() );
            gpuErrchk( cudaFree(d_pwv_screen) );

            idx_wrap++;
        }
        printf("*** Progress: 100 / 100 ***\r");
    }
    gpuErrchk( cudaDeviceReset() );
    printf("\033[0m\n");
}

