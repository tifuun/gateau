cmake_minimum_required(VERSION 3.16.3)
 
# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()

include(GoogleTest)

find_library(INTERPLIB
      NAMES interputils
      PATHS tiempo2)

  # Link runTests with what we want to test and the GTest and pthread library
add_executable(runTests include/tests/test_InterpUtils.cpp)
target_link_libraries(runTests ${INTERPLIB} ${GTEST_LIBRARIES} GTest::gtest_main)
# target_link_libraries(runTests ${GTEST_LIBRARIES} GTest::gtest_main)
set_target_properties(runTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
if(NOT WIN32)
    target_link_libraries(runTests pthread)
endif()
gtest_discover_tests(runTests)

