cmake_minimum_required(VERSION 3.16.3)

project(gateau 
	VERSION 0.0.1 
	DESCRIPTION "gateau: GPU-Accelerated Time-dEpendent observAtion simUlator"
	LANGUAGES CXX C)

option(BUILD_TESTS "Build unit tests with Google Test" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
find_package(GSL REQUIRED)

include(CheckLanguage)

check_language(CUDA)

if(NOT BUILD_TESTS)
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

else()
	enable_testing()

	enable_language(CUDA)

	file(GLOB CUDAfiles src/gateau/cuda/*.cu)

	#add_library(gateau SHARED ${CUDAfiles})
	add_executable(gateau_tests gtest/testBasic.cu )#${CUDAfiles})
	target_include_directories(gateau_tests PRIVATE src/gateau/cuda src/gateau/include)
	set_target_properties(gateau_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_property(TARGET gateau_tests PROPERTY CUDA_ARCHITECTURES OFF)
	if(TARGET GSL::gsl)
		target_link_libraries(gateau_tests PRIVATE GSL::gsl gtest gtest_main)
	else()
		target_include_directories(gateau_tests PRIVATE ${GSL_INCLUDE_DIRS})
		target_link_libraries(gateau_tests PRIVATE ${GSL_LIBRARIES} gtest gtest_main)
	endif()

	#target_link_libraries(gateau_tests gtest_main)

    include(GoogleTest)
    gtest_discover_tests(gateau_tests)

endif()

