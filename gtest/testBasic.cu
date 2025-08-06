/*
 * Basic tests for Gateau CPP code
 * TODO doxy comments!
 */

#include <regex>
#include <gtest/gtest.h>
#include "../src/gateau/include/structs.h"
#include "../src/gateau/include/utilities.h"
#include "../src/gateau/include/fio.h"
// TODO path traversal okay like that?? I'm not a C++ dev

TEST(BasicTest, ReadEtaATMTest) {
    ArrSpec pwv_atm = {
        0.0,  // start
        0.1,  // step
        2     // num
    };
    ArrSpec freq_atm = {
        0.0,  // start
        0.1,  // step
        2     // num
    };
    float **eta_arr;

    eta_arr = (float**)malloc(sizeof(float*) * 2);
    for (int i = 0; i < 2; i++) {
        eta_arr[i] = (float*)malloc(sizeof(float) * 2);
    }

    std::regex target(R"(gtest\/testBasic\.cu$)");
    std::string rel_loc = "src/gateau/resources";
    std::string abs_loc = std::regex_replace(__FILE__, target, rel_loc);

    readEtaATM(
        eta_arr,
        &pwv_atm,
        &freq_atm,
        abs_loc.c_str()
        );
}

TEST(BasicTest, ReadEtaATMTestWithTrailingSlash) {
    ArrSpec pwv_atm = {
        0.0,  // start
        0.1,  // step
        2     // num
    };
    ArrSpec freq_atm = {
        0.0,  // start
        0.1,  // step
        2     // num
    };
    float **eta_arr;

    eta_arr = (float**)malloc(sizeof(float*) * 2);
    for (int i = 0; i < 2; i++) {
        eta_arr[i] = (float*)malloc(sizeof(float) * 2);
    }

    std::regex target(R"(gtest\/testBasic\.cu$)");
    std::string rel_loc = "src/gateau/resources/";
    std::string abs_loc = std::regex_replace(__FILE__, target, rel_loc);

    readEtaATM(
        eta_arr,
        &pwv_atm,
        &freq_atm,
        abs_loc.c_str()
        );
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
