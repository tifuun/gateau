#include <gtest/gtest.h>
#include "../InterpUtils.h"
#include <iostream>
#include <string>
#include <fstream>

double func(double xt, double yt, int ax_to_change);
class InterpUtilsTest : public ::testing::Test {
    protected:

    // Make a grid on which to test interpolation
    double err = 1e-12;

    int size_x = 7;
    int size_y = 5;
    double x[7] = {-2, -1, 0, 1, 2, 3, 4};
    double y[5] = {-1, 0, 1, 2, 3};
    
    int size_arr = 35;
    double arrx[35];
    double arry[35];

    int size_arr_to_test = 10000;
    double arr_to_testx[10000];
    double arr_to_testy[10000];

    int size_interp = 100;
    double x_i[100];
    double y_i[100];
    int ax_to_change = 0;
    
    void SetUp() override {
        for(int i=0; i<size_x; i++) {
            for(int j = 0; j<size_y; j++) {
                this->arrx[i * size_y + j] = func(this->x[i], this->y[j], ax_to_change);
                ax_to_change = 1;
                this->arry[i * size_y + j] = func(this->x[i], this->y[j], ax_to_change);
            }
            //printf("%f, %f, %f, %f, %f\n", this->arr[i * size_y], this->arr[i * size_y+1], this->arr[i * size_y+2], this->arr[i * size_y+3], this->arr[i * size_y+4] );
        }

        std::string to_test_s;
        
        std::ifstream read_x("include/tests/x_i.test");
        std::ifstream read_y("include/tests/y_i.test");
        
        int n = 0;
        while (read_x >> to_test_s) {
            double x = atof(to_test_s.c_str());
            x_i[n] = x; 
            n++;
        }

        n = 0;
        
        while (std::getline (read_y, to_test_s)) {
            y_i[n] = atof(to_test_s.c_str());
            n++;
        }

        for(int i=0; i<size_interp; i++) {
            for(int j = 0; j<size_interp; j++) {
                ax_to_change = 0;
                this->arr_to_testx[i * size_interp + j] = func(this->x_i[i], this->y_i[j], ax_to_change);
                ax_to_change = 1;
                this->arr_to_testy[i * size_interp + j] = func(this->x_i[i], this->y_i[j], ax_to_change);
                
            }
        }
    }
};

/**
 * Function to test the linear interpolation.
 * This function changes slope at certain points for x and y, to see if the interpolation can track this.
 *
 * @param x Value for x.
 * @param y Value for y.
 */
double func(double xt, double yt, int ax_to_change) {
    double start = 1.414;

    double slope_x = 1 / 3.142;
    double slope_y = -2.718;


    double out = slope_x * xt + slope_y * yt + start;
    return out;
}


TEST_F(InterpUtilsTest, TestInterpValue) {
    double cpp_val_x;
    double cpp_val_y;
    bool debug = false;

    int nerr = 0;
    for(int i=0; i<85; i++) {
        for(int j=0; j<2; j++) {
            cpp_val_x = interpValue(x_i[i], y_i[j], x, y, size_x, size_y, arrx, debug);
            cpp_val_y = interpValue(x_i[i], y_i[j], x, y, size_x, size_y, arry, debug);
            //printf("%f, %f\n", cpp_val, arr_to_test[i * size_interp + j]);
            EXPECT_NEAR(cpp_val_x, arr_to_testx[i * size_interp + j], err);
            EXPECT_NEAR(cpp_val_y, arr_to_testy[i * size_interp + j], err);
        }
    }
}

