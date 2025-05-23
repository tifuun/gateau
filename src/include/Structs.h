/*!
 * \file
 * \brief Data structures for receiving data from Python interface.
 **/

#ifndef __Structs_h
#define __Structs_h

template<typename T>
struct Instrument;

template<typename T>
struct Telescope;

template<typename T>
struct Atmosphere;

template<typename T>
struct Source;

template<typename T>
struct Cascade;

template<typename T>
struct CalOutput;

template<typename T>
struct ArrSpec;

template<typename T>
struct ArrSpec {
    T start;
    T step;
    int num;
};

template<typename T>
struct Cascade {
    T *eta; /**< Efficiency term associated with this cascade.*/
    T *T_parasitic; /**< Noise temperature of parasitic source.*/
    T *d; /**< Thickness of window/lens in meters.*/
    T *tandelta; /**< Loss tangent of dielectric.*/
    T *neff; /**< Effective dielectric constant.*/
    T *T_parasitic_refl; /**< Temperature of parasitic source seen in reflection.*/ 
    T *T_parasitic_refr; /**< Temperature of parasitic source seen in refraction.*/
    int *use_AR; /**< Whether to use AR coating or not.*/
    int *order_refl; /** Array with indices of elements from the reflect array in the total cascade.*/
    int *order_refr; /** Array with indices of elements from the refract array in the total cascade.*/
};

// Ctypes structs - for communicating with python
template<typename T>
struct Instrument {
    int nf_ch;          /**< Number of elements in freqs.*/
    
    struct ArrSpec<T> f_spec;

    //T R;            /**< Resolving power.*/
    //T *f_ch;        /**< Array with channel frequencies.*/
    //int order;      /**< Order of Lorentzian filters.*/
    
    //T *eta_filt;     /**<Peak height of filter, for each channel.*/
    T f_sample; /**< Readout frequency of instrument in Hertz.*/
    T *filterbank; /**< Array with filterbank matrix, flattened.*/
    T delta;       /**< Superconducting bandgap energy in Joules.*/
    T eta_pb;      /**< Pair breaking efficiency of superconductor.*/
};

template<typename T>
struct Telescope {
    T Dtel;        /**< Primary aperture diameter in meters.*/
    int chop_mode;      /**< Chopping mode. 0 is 'none', 1 is 'direct', 2 is 'abba'.*/
    T dAz_chop;    /**< Azimuthal separation between chopping paths.*/
    T freq_chop;   /**< Chopping frequency in Hertz. If < 0, no chopping.*/
    T freq_nod;    /**< Nodding frequency in Hertz.*/
    T *eta_ap_ON;  /**< Array of aperture efficiencies in ON position, as function of frequency (set by Instrument). Size is nfreqs of instrument.*/
    T *eta_ap_OFF; /**< Array of aperture efficiencies in OFF position, as function of frequency (set by Instrument). Size is nfreqs of instrument.*/
    int scantype;
    T El0;
    T Ax;
    T Axmin;
    T Ay;
    T Aymin;
    T wx;
    T wxmin;
    T wy;
    T wymin;
    T phix;
    T phiy;
};

template<typename T>
struct Atmosphere {
    T Tatm;        /**< Temperature of atmosphere in Kelvin.*/
    T v_wind;      /**< Max windspeed in meters per second.*/
    T h_column;    /**< Reference column height of atmosphere, in meters.*/
    T dx;          /**< Gridsize along x axis in meters.*/
    T dy;          /**< Gridsize along y axis in meters.*/
    char* path;    /**< Path to prepd folder.*/
};

template<typename T>
struct Source {
    struct ArrSpec<T> Az_spec;
    struct ArrSpec<T> El_spec;
    
    T *I_nu;       /**< Flat array of specific intensities.*/
    int nI_nu;         /**< Number of source intensities.*/
};

template<typename T>
struct CalOutput {
    T *power;       /**< Power in Watt as function of filter index (axis 0) and PWV (axis 1).*/
    T *temperature; /**< LOS brightness temperature in Kelvin as function of filter index (axis 0) and PWV (axis 1).*/
};

// Local structs - for use internally
template<typename T>
struct Effs {
    T eta_tot_chain; /**< Total constant efficiency after atmosphere.*/
    T eta_tot_gnd;   /**< Total constant efficiency after groundi, including ground emissivity.*/
    T eta_tot_mir;   /**< Total constant efficiency after mirrors, including mirror emissivity.*/
};

#endif
