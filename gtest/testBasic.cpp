/*
 * Basic tests for Gateau CPP code
 * TODO doxy comments!
 */

#include <gtest/gtest.h>

TEST(BasicTest, ReadEtaATMTest) {
	//int eta_array = NULL;
	//readEtaATM(T **eta_array, U *pwv_atm, U *freq_atm)
	EXPECT_EQ(1, 2);
}

int main(int argc, char* argv[]) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
