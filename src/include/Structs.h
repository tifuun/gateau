/*!
 * \file
 * \brief Data structures for receiving data from Python interface.
 **/

#ifndef __Structs_h
#define __Structs_h

struct Instrument;
struct Telescope;
struct atmosphere;
struct Source;
struct Cascade;
struct ArrSpec;

struct ArrSpec {
    float start;
    float step;
    int num;
};

struct Cascade {
    float *eta_stage; /**< Efficiency terms associated with each stage of this cascade.*/
    float *psd_stage; /**< Power spectral density of each parasitic source of this cascade.*/
    int num_stage; /**< Number of (grouped) stages in cascade.*/
};

struct Instrument {
    int nf_ch;          /**< Number of elements in freqs.*/
    struct ArrSpec f_spec;
    float f_sample; /**< Readout frequency of instrument in Hertz.*/
    float *filterbank; /**< Array with filterbank matrix, flattened.*/
    float delta;       /**< Superconducting bandgap energy in Joules.*/
    float eta_pb;      /**< Pair breaking efficiency of superconductor.*/
};

struct Telescope {
    float *eta_ap;      /**< Array of aperture efficiencies.*/
    float *az_scan;     /**< Azimuth co-ordinates of scan strategy for simulation.*/
    float *el_scan;     /**< Elevation co-ordinates of scan strategy for simulation.*/
};


struct Atmosphere {
    float T_atm;        /**< floatemperature of atmosphere in Kelvin.*/
    float v_wind;      /**< Max windspeed in meters per second.*/
    float h_column;    /**< Reference column height of atmosphere, in meters.*/
    float dx;          /**< Gridsize along x axis in meters.*/
    float dy;          /**< Gridsize along y axis in meters.*/
    char* path;    /**< Path to prepd folder.*/
};


struct Source {
    struct ArrSpec az_src_spec;
    struct ArrSpec el_src_spec;
    
    float *I_nu;       /**< Flat array of specific intensities.*/
    int nI_nu;         /**< Number of source intensities.*/
};

#endif
