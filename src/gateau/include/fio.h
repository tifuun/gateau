/*! \file FileIO.h
 * \brief File input/output operations.
 **/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
//#include <cxxabi.h> THIS IS NOT PRESENT ON MICROSHIT CPP COMPILER
#include <filesystem>

#include "hdf5.h"

#include "structs.h"

#ifndef __FILEIO_H
#define __FILEIO_H

namespace fs = std::filesystem;

#define NPWVATM         55
#define NFREQ           8301
#define NATMGRID        3
#define COLBUFF         16
#define FMTBUFF         4

#define OBSATTRS_NAME   "OBSATTRS"
#define SPAX_NAME       "SPAXEL"
#define FREQS_NAME      "frequencies"
#define TIME_NAME       "times"
#define AZ_NAME         "az"
#define EL_NAME         "el"

#define AZ_SPAX_NAME    "az_spax"
#define EL_SPAX_NAME    "el_spax"
#define OUT_NAME        "data"

#define CHBUFF          100
#define RANK1D          1
#define RANK2D          2

void readAtmMeta(int **meta, std::string path);

template <typename T, typename U>
void readEtaATM(
        T **eta_array, 
        U *pwv_atm, 
        U *freq_atm);

template <typename T, typename U>
void readAtmScreen(
        T **PWV_screen, 
        U *x_spec, 
        U *y_spec, 
        std::string path, 
        std::string datp); 

template <typename T>
void write1DArray(
        T *array, 
        int narr, 
        std::string path, 
        std::string name);

class OutputFile {
    private:
        // Handles for hdf5 file and groups
        hid_t   file_id;
        hid_t   obsattrs_id;
        hid_t   spax_id;
        
        // Handles for hdf5 dataspace/set creation
        hid_t   dspace_id, dspace_slab_id, dset_id;

        // Output dimensions
        hsize_t dims_1D[RANK1D], 
                dims_2D[RANK2D], 
                dims_2D_chunk[RANK2D],
                dims_2D_stride[RANK2D];

        hsize_t dims_2D_null[2] = {0,0};

        // Hyperslab dimensions
        hsize_t start[2], count[2];

        int ntimes, nfreqs;
        int offset_times = 0;

        void check_API_call_status(
                herr_t status, 
                int loc)
        {
            if(status < 0)
            {
                printf("HDF5 API error occured on line %d\n", loc);
            }
        }
    
    public:
        OutputFile(
                const char *filename, 
                int ntimes, 
                int nfreqs, 
                float *freqs, 
                float *times, 
                float *az, 
                float *el) 
        {
            this->ntimes = ntimes;
            this->nfreqs = nfreqs;

            dims_2D[0] = ntimes;
            dims_2D[1] = nfreqs;

            start[1] = 0;
            count[1] = nfreqs;

            // Make file and obsattrs group
            file_id = H5Fcreate(
                    filename, 
                    H5F_ACC_TRUNC, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT);
            
            obsattrs_id = H5Gcreate(
                    file_id, 
                    OBSATTRS_NAME, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT);   

            // Write dataspace for frequencies
            dims_1D[0] = nfreqs;
            
            dspace_id = H5Screate_simple(
                    RANK1D, 
                    dims_1D, 
                    NULL);

            dset_id = H5Dcreate(
                    obsattrs_id, 
                    FREQS_NAME, 
                    H5T_NATIVE_FLOAT, 
                    dspace_id, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT);

            check_API_call_status( 
                    H5Dwrite(
                        dset_id, 
                        H5T_NATIVE_FLOAT, 
                        H5S_ALL, 
                        H5S_ALL, 
                        H5P_DEFAULT, 
                        freqs
                        ),
                    __LINE__
                    );

            // Initialise time and az-el arrays
            dims_1D[0] = ntimes;

            dspace_id = H5Screate_simple(
                    RANK1D, 
                    dims_1D, 
                    NULL);
            
            dset_id = H5Dcreate(
                    obsattrs_id, 
                    TIME_NAME, 
                    H5T_NATIVE_FLOAT, 
                    dspace_id, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT);
            
            check_API_call_status(
                    H5Dwrite(
                        dset_id, 
                        H5T_NATIVE_FLOAT, 
                        H5S_ALL, 
                        H5S_ALL, 
                        H5P_DEFAULT, 
                        times
                        ),
                    __LINE__
                    );

            dset_id = H5Dcreate(
                    obsattrs_id, 
                    AZ_NAME, 
                    H5T_NATIVE_FLOAT, 
                    dspace_id, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT);

            check_API_call_status(
                    H5Dwrite(
                        dset_id, 
                        H5T_NATIVE_FLOAT, 
                        H5S_ALL, 
                        H5S_ALL, 
                        H5P_DEFAULT, 
                        az
                        ),
                    __LINE__
                    );

            dset_id = H5Dcreate(
                    obsattrs_id, 
                    EL_NAME, 
                    H5T_NATIVE_FLOAT, 
                    dspace_id, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT);
            
            check_API_call_status(
                    H5Dwrite(
                        dset_id, 
                        H5T_NATIVE_FLOAT, 
                        H5S_ALL, 
                        H5S_ALL, 
                        H5P_DEFAULT, 
                        el
                        ),
                    __LINE__
                    );

            check_API_call_status(
                    H5Sclose(dspace_id),
                    __LINE__
                    );
            check_API_call_status(
                    H5Dclose(dset_id),
                    __LINE__
                    );
        }

        void open_spaxel(
                int spax_index,
                float az_spax, 
                float el_spax) 
        {
            offset_times = 0;
            
            char spax_name[CHBUFF] = SPAX_NAME;
            char buffer[CHBUFF];
            
            sprintf(
                    buffer, 
                    "%d", 
                    spax_index);
            
            strcat(
                    spax_name, 
                    buffer);
            
            spax_id = H5Gcreate(
                    file_id, 
                    spax_name, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT, 
                    H5P_DEFAULT);

            dspace_id = H5Screate(H5S_SCALAR);

            dset_id = H5Dcreate(
                    spax_id,
                    AZ_SPAX_NAME,
                    H5T_NATIVE_FLOAT,
                    dspace_id,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT);

            check_API_call_status(
                    H5Dwrite(
                        dset_id, 
                        H5T_NATIVE_FLOAT, 
                        H5S_ALL, 
                        H5S_ALL, 
                        H5P_DEFAULT, 
                        &az_spax
                        ),
                    __LINE__
                    );
            
            dset_id = H5Dcreate(
                    spax_id,
                    EL_SPAX_NAME,
                    H5T_NATIVE_FLOAT,
                    dspace_id,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT);

            check_API_call_status(
                    H5Dwrite(
                        dset_id, 
                        H5T_NATIVE_FLOAT, 
                        H5S_ALL, 
                        H5S_ALL, 
                        H5P_DEFAULT, 
                        &el_spax
                        ),
                    __LINE__
                    );

            // Allocate output array for this spaxel
            dspace_id = H5Screate_simple(
                    RANK2D, 
                    dims_2D, 
                    NULL);
            
            dset_id = H5Dcreate(
                    spax_id,
                    OUT_NAME,
                    H5T_NATIVE_FLOAT,
                    dspace_id,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT);
        }

        void write_chunk_to_spaxel(
                int ntimes_chunk, 
                float *data)
        {
            start[0] = offset_times;
            count[0] = ntimes_chunk;

            dims_1D[0] = ntimes_chunk * nfreqs;

            check_API_call_status(
                    H5Sselect_hyperslab(
                        dspace_id,
                        H5S_SELECT_SET,
                        start,
                        NULL,
                        count,
                        NULL
                        ),
                    __LINE__
                    );

            dspace_slab_id = H5Screate_simple(
                    RANK1D,
                    dims_1D,
                    NULL);
            
            check_API_call_status(
                    H5Dwrite(
                        dset_id,
                        H5T_NATIVE_FLOAT,
                        dspace_slab_id,
                        dspace_id,
                        H5P_DEFAULT,
                        data
                        ),
                    __LINE__
                    );
            

            offset_times += ntimes_chunk;
            check_API_call_status(
                    H5Sclose(dspace_slab_id),
                    __LINE__
                    );
        }

        void close_spaxel(int spax_index) 
        {
            check_API_call_status(
                    H5Sclose(dspace_id),
                    __LINE__
                    );
            check_API_call_status(
                    H5Dclose(dset_id),
                    __LINE__
                    );
            check_API_call_status(
                    H5Gclose(spax_id),
                    __LINE__
                    );
        }

        ~OutputFile() 
        {
            check_API_call_status(
                    H5Fclose(file_id),
                    __LINE__
                    );
        }
};

#endif

void readAtmMeta(
        int **meta, 
        std::string path) 
{
    fs::path dir(path);
    fs::path file("atm_meta.datp");
    fs::path abs_loc = dir / file;

    *meta = new int[NATMGRID];
    
    std::string store;

    std::ifstream myfile(abs_loc);
    std::string line;

    int idx = 0;
    
    if (!myfile) 
    {
    	std::cerr
            << "Could not open the file at "
            << abs_loc
            << std::endl;
	    exit(5);
    }
    else 
    {
        while(std::getline(myfile, line)) 
        {
            std::istringstream iss(line);
            while(std::getline(iss, store, ' ')) 
            {
                if (store=="") {continue;}
                (*meta)[idx] = std::stoi(store);
                idx++;
            }
        }
        myfile.close();
    }
}

template <typename T, typename U>
void readEtaATM(
        T **eta_array,
        U *pwv_atm,
        U *freq_atm,
        const char* filepath
        ) 
{    
    // TODO read these in from file? Ask Arend
    pwv_atm->start = 0.1;
    pwv_atm->step = 0.1;
    pwv_atm->num = NPWVATM;

    freq_atm->start = 70.e9;
    freq_atm->step = 0.1e9;
    freq_atm->num = NFREQ;

    *eta_array = new T[NPWVATM * NFREQ];
    
    std::string store;
    //std::cout << abi::__cxa_demangle(typeid(store).name(), NULL, NULL, &status) << std::endl;

    std::ifstream myfile(filepath);
    std::string line;

    int line_nr = 0;
    int idx = 0;
    
    if (!myfile) 
    {
	    std::cerr
		    << "Could not open the resource file at "
		    << filepath
		    << std::endl;
	    exit(5);
	    /* TODO Standardize exit codes */
	}

    while(std::getline(myfile, line)) 
    {
        std::istringstream iss(line);
        if(!line_nr) 
        {
            line_nr++;
            continue;
        } 
        
        while(std::getline(iss, store, ' ')) 
        {
            if(!idx) 
            {
                idx++;
                continue;
            } 
            else if (store=="") 
            {
                continue;
            }
            
            (*eta_array)[NFREQ * (idx-1) + (line_nr - 1)] = std::stof(store);
            idx++;
        }
        line_nr++;
        idx = 0;
    }
    myfile.close();
}

template <typename T, typename U>
void readAtmScreen(
        T **PWV_screen, 
        U *x_spec, 
        U *y_spec, 
        std::string path, 
        std::string datp) 
{
    fs::path dir(path);
    fs::path file(datp);
    fs::path abs_loc = dir / file;

    *PWV_screen = new T[x_spec->num * y_spec->num];
    
    std::string store;

    std::ifstream myfile(abs_loc);
    std::string line;

    int line_nr = 0;
    int idx = 0;
    
    if (!myfile)
    { 
        std::cerr 
            << "Could not open the file!" 
            << std::endl;
    }
    else
    {
        while(std::getline(myfile, line)) 
        {
            std::istringstream iss(line);
            while(std::getline(iss, store, ' ')) 
            {
                if (store=="") 
                {
                    continue;
                }
                
                (*PWV_screen)[y_spec->num * line_nr + idx] = std::stof(store);
                idx++;
            }
            line_nr++;
            idx = 0;
        }
        myfile.close();
    }
}

template <typename T>
void write1DArray(std::vector<T> array, std::string path, std::string name, std::string subpath = "") {
    fs::path dir(path);
    fs::path subdir(subpath);
    fs::path file(name);

    fs::path abs_loc = dir / subdir / file;
    std::ofstream myfile (abs_loc, std::ios::binary | std::ios::trunc);
    myfile.write(reinterpret_cast<const char*>(array.data()), array.size() * sizeof(T));
    myfile.close();
}
