cmake_minimum_required(VERSION 3.16.3)

project(gateau 
    VERSION 0.0.1 
    DESCRIPTION "gateau: GPU-Accelerated Time-dEpendent observAtion simUlator"
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(timer SHARED src/include/Timer.cpp)
target_include_directories(timer PUBLIC src/include)

include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    file(GLOB CUDAfiles src/cuda/*.cu)

    add_library(cugateau SHARED ${CUDAfiles})
    add_library(cugateauptx OBJECT ${CUDAfiles})
    target_include_directories(cugateau PRIVATE src/cuda src/include)
    target_link_libraries(cugateau PRIVATE timer)
    set_target_properties(cugateau PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET cugateau PROPERTY CUDA_ARCHITECTURES OFF)
    
    target_include_directories(cugateauptx PRIVATE src/cuda src/include)
    target_link_libraries(cugateauptx PRIVATE timer)
    set_property(TARGET cugateauptx PROPERTY CUDA_PTX_COMPILATION ON)
else()
    message(STATUS "WARNING: no CUDA compiler detected. Not building CUDA libraries")
endif()
