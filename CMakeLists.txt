cmake_minimum_required(VERSION 3.16.3)

project(gateau 
	VERSION 0.0.1 
	DESCRIPTION "gateau: GPU-Accelerated Time-dEpendent observAtion simUlator"
	LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
find_package(GSL REQUIRED)

include(CheckLanguage)

check_language(CUDA)

# FIXME is this right?? Project name??
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SKBUILD_PLATLIB_DIR}/${SKBUILD_PROJECT_NAME}")

if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	file(GLOB CUDAfiles src/cuda/*.cu)

	add_library(gateau SHARED ${CUDAfiles})
	target_include_directories(gateau PRIVATE src/cuda src/include)
	set_target_properties(gateau PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_property(TARGET gateau PROPERTY CUDA_ARCHITECTURES OFF)
	if(TARGET GSL::gsl)
		target_link_libraries(gateau PRIVATE GSL::gsl)
	else()
		target_include_directories(gateau PRIVATE ${GSL_INCLUDE_DIRS})
		target_link_libraries(gateau PRIVATE ${GSL_LIBRARIES})
	endif()
else()
    message(FATAL_ERROR "FATAL ERROR: no CUDA compiler detected. Not building GATEAU.")
endif()


