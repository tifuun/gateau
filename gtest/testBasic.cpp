/*
 * Basic tests for Gateau CPP code
 * TODO doxy comments!
 */

#include <gtest/gtest.h>
#include <gtest/gtest.h>
#include "../src/gateau/include/structs.h"
#include "../src/gateau/include/utilities.h"
// TODO path traversal okay like that?? I'm not a C++ dev

TEST(BasicTest, ReadEtaATMTest) {
	float eta_arr[2][2] = {
		{0.0, 0.0},
		{0.0, 0.0},
		};
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

	readEtaATM(eta_arr, &pwv_atm, &freq_atm)
}

int main(int argc, char* argv[]) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
