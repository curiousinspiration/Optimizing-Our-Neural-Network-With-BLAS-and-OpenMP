/*
 * ReLU Layer Test
 *
 */

#include "neural/layers/relu_layer.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(ReLULayerTest, TestForward)
{
        // Init input
    TTensorPtr input = Tensor::New({2,2}, {
        -4.0, 3.0,
        2.0, -1.0
    });

    ReLULayer layer;

    // Relu layer should calculate max(0,x) for each value in the matrix
    TTensorPtr output = layer.Forward(input);
    EXPECT_EQ(2, output->Shape().at(0));
    EXPECT_EQ(2, output->Shape().at(1));

    /*
    (0,0) = max(0.0, -4.0) = 0
    (0,1) = max(0.0,  3.0) = 3.0
    (1,0) = max(0.0,  2.0) = 2.0
    (1,1) = max(0.0, -1.0) = 0
    */

    EXPECT_EQ(0.0, output->At({0,0}));
    EXPECT_EQ(3.0, output->At({0,1}));
    EXPECT_EQ(2.0, output->At({1,0}));
    EXPECT_EQ(0.0, output->At({1,1}));
}
