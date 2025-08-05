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

if("${SKBUILD_STATE}" STREQUAL "editable")
	set(
		CMAKE_LIBRARY_OUTPUT_DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}/src/${SKBUILD_PROJECT_NAME}"
	)
else()
	set(
		CMAKE_LIBRARY_OUTPUT_DIRECTORY
		"${SKBUILD_PLATLIB_DIR}/${SKBUILD_PROJECT_NAME}"
	)
endif()

if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	file(GLOB CUDAfiles src/gateau/cuda/*.cu)

	add_library(gateau SHARED ${CUDAfiles})
	target_include_directories(gateau PRIVATE src/gateau/cuda src/gateau/include)
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

## Okay I have no clue what I am doing here!!

enable_testing()

enable_language(CUDA)

add_executable(
	gateau_test
	gtest/testBasic.cpp
)
target_include_directories(gateau_test
	PUBLIC
	src/gateau/cuda 
        src/gateau/include
)
target_include_directories(gateau_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(gateau_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(gateau_test PRIVATE gateau gtest gtest_main)

set_property(TARGET gateau_test PROPERTY CUDA_ARCHITECTURES OFF)

include(GoogleTest)
gtest_discover_tests(gateau_test)


