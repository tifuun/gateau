/*! \file FileIO.h
 * \brief File input/output operations.
 **/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>
#include <cxxabi.h>
#include <filesystem>
#include "Structs.h"

#ifndef __FILEIO_H
#define __FILEIO_H

namespace fs = std::filesystem;

#define NPWV    56
#define NFREQ   8301
#define NATMGRID 3

void readAtmMeta(int **meta, std::string path);

template <typename T, typename U>
void readEtaATM(T **eta_array, U *pwv_atm, U *freq_atm);

template <typename T, typename U>
void readAtmScreen(T **PWV_screen, U *x_spec, U *y_spec, std::string path, std::string datp); 

template <typename T>
void write1DArray(T *array, int narr, std::string path, std::string name);

#endif

void readAtmMeta(int **meta, std::string path) {
    fs::path dir(path);
    fs::path file("atm_meta.datp");
    fs::path abs_loc = dir / file;

    *meta = new int[NATMGRID];
    
    std::string store;

    std::ifstream myfile(abs_loc);
    std::string line;

    int idx = 0;
    
    if (!myfile) std::cerr << "Could not open the file!" << std::endl;
    else {
        while(std::getline(myfile, line)) {
            std::istringstream iss(line);
            while(std::getline(iss, store, ' ')) {
                if (store=="") {continue;}
                (*meta)[idx] = std::stoi(store);
                idx++;
            }
        }
        myfile.close();
    }
}

template <typename T, typename U>
void readEtaATM(T **eta_array, U *pwv_atm, U *freq_atm) {
    
    pwv_atm->start = 0.1;
    pwv_atm->step = 0.1;
    pwv_atm->num = NPWV;

    freq_atm->start = 70.e9;
    freq_atm->step = 0.1e9;
    freq_atm->num = NFREQ;

    std::regex target("include\/FileIO\.h");
    std::string rel_loc = "resources/eta_atm";
    std::string abs_loc = std::regex_replace(__FILE__, target, rel_loc);

    *eta_array = new T[NPWV * NFREQ];
    
    std::string store;
    //std::cout << abi::__cxa_demangle(typeid(store).name(), NULL, NULL, &status) << std::endl;

    std::ifstream myfile(abs_loc);
    std::string line;

    int line_nr = 0;
    int idx = 0;
    
    if (!myfile) std::cerr << "Could not open the file!" << std::endl;
    else {
        while(std::getline(myfile, line)) {
            std::istringstream iss(line);
            if(!line_nr) {
                line_nr++;
                continue;
            } 
            
            while(std::getline(iss, store, ' ')) {
                if(!idx) {
                    idx++;
                    continue;
                } else if (store=="") {continue;}
                //std::cout << abi::__cxa_demangle(typeid(store).name(), NULL, NULL, &status) << std::endl;
                //iss.ignore();
                (*eta_array)[NFREQ * (idx-1) + (line_nr - 1)] = std::stof(store);
                idx++;
                
            }
            line_nr++;
            idx = 0;
        }
        myfile.close();
        //std::cout << "here" << std::endl;
    }
}

template <typename T, typename U>
void readAtmScreen(T **PWV_screen, U *x_spec, U *y_spec, std::string path, std::string datp) {
    fs::path dir(path);
    fs::path file(datp);
    fs::path abs_loc = dir / file;

    *PWV_screen = new T[x_spec->num * y_spec->num];
    
    std::string store;

    std::ifstream myfile(abs_loc);
    std::string line;

    int line_nr = 0;
    int idx = 0;
    
    if (!myfile) std::cerr << "Could not open the file!" << std::endl;
    else {
        while(std::getline(myfile, line)) {
            std::istringstream iss(line);
            while(std::getline(iss, store, ' ')) {
                if (store=="") {continue;}
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
void write1DArray(std::vector<T> array, std::string path, std::string name) {
    fs::path dir(path);
    fs::path file(name);
    fs::path abs_loc = dir / file;
    std::ofstream myfile (abs_loc, std::ios::binary | std::ios::trunc);
    myfile.write(reinterpret_cast<const char*>(array.data()), array.size() * sizeof(T));
    myfile.close();
}
